package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha1"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"image"
	"image/color"
	"io"
	"io/fs"
	"log"
	"math"
	"math/bits"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/mattn/go-sqlite3"
	"golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

// #include <unistd.h>
import "C"

var (
	db               *sql.DB // sqlite database
	galleryDirectory string  // gallery directory

	// taskSemaphore limits parallel computations.
	taskSemaphore semaphore
)

const (
	nameOfDB        = "gallery.db"
	nameOfImageRoot = "images"
	nameOfThumbRoot = "thumbs"
)

func hammingDistance(a, b int64) int {
	return bits.OnesCount64(uint64(a) ^ uint64(b))
}

type productAggregator float64

func (pa *productAggregator) Step(v float64) {
	*pa = productAggregator(float64(*pa) * v)
}

func (pa *productAggregator) Done() float64 {
	return float64(*pa)
}

func newProductAggregator() *productAggregator {
	pa := productAggregator(1)
	return &pa
}

func init() {
	sql.Register("sqlite3_custom", &sqlite3.SQLiteDriver{
		ConnectHook: func(conn *sqlite3.SQLiteConn) error {
			if err := conn.RegisterFunc(
				"hamming", hammingDistance, true /*pure*/); err != nil {
				return err
			}
			if err := conn.RegisterAggregator(
				"product", newProductAggregator, true /*pure*/); err != nil {
				return err
			}
			return nil
		},
	})
}

func openDB(directory string) error {
	var err error
	db, err = sql.Open("sqlite3_custom", "file:"+filepath.Join(directory,
		nameOfDB+"?_foreign_keys=1&_busy_timeout=1000"))
	galleryDirectory = directory
	return err
}

func imagePath(sha1 string) string {
	return filepath.Join(galleryDirectory,
		nameOfImageRoot, sha1[:2], sha1)
}

func thumbPath(sha1 string) string {
	return filepath.Join(galleryDirectory,
		nameOfThumbRoot, sha1[:2], sha1+".webp")
}

func dbCollectStrings(query string, a ...any) ([]string, error) {
	rows, err := db.Query(query, a...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := []string{}
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			return nil, err
		}
		result = append(result, s)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

func idForDirectoryPath(tx *sql.Tx, path []string, create bool) (int64, error) {
	var parent sql.NullInt64
	for _, name := range path {
		if err := tx.QueryRow(`SELECT id FROM node
			WHERE parent IS ? AND name = ? AND sha1 IS NULL`,
			parent, name).Scan(&parent); err == nil {
			continue
		} else if !errors.Is(err, sql.ErrNoRows) {
			return 0, err
		} else if !create {
			return 0, err
		}

		// This fails when trying to override a leaf node.
		// That needs special handling.
		if result, err := tx.Exec(
			`INSERT INTO node(parent, name) VALUES (?, ?)`,
			parent, name); err != nil {
			return 0, err
		} else if id, err := result.LastInsertId(); err != nil {
			return 0, err
		} else {
			parent = sql.NullInt64{Int64: id, Valid: true}
		}
	}
	return parent.Int64, nil
}

func decodeWebPath(path string) []string {
	// Relative paths could be handled differently,
	// but right now, they're assumed to start at the root.
	result := []string{}
	for _, crumb := range strings.Split(path, "/") {
		if crumb != "" {
			result = append(result, crumb)
		}
	}
	return result
}

// --- Semaphore ---------------------------------------------------------------

type semaphore chan struct{}

func newSemaphore(size int) semaphore { return make(chan struct{}, size) }
func (s semaphore) release()          { <-s }

func (s semaphore) acquire(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case s <- struct{}{}:
	}

	// Give priority to context cancellation.
	select {
	case <-ctx.Done():
		s.release()
		return ctx.Err()
	default:
	}
	return nil
}

// --- Progress bar ------------------------------------------------------------

type progressBar struct {
	sync.Mutex
	current int
	target  int
}

func newProgressBar(target int) *progressBar {
	pb := &progressBar{current: 0, target: target}
	pb.Update()
	return pb
}

func (pb *progressBar) Stop() {
	// The minimum thing that works: just print a newline.
	os.Stdout.WriteString("\n")
}

func (pb *progressBar) Update() {
	if pb.target < 0 {
		fmt.Printf("\r%d/?", pb.current)
		return
	}

	var fraction int
	if pb.target != 0 {
		fraction = int(float32(pb.current) / float32(pb.target) * 100)
	}

	target := fmt.Sprintf("%d", pb.target)
	fmt.Printf("\r%*d/%s (%2d%%)", len(target), pb.current, target, fraction)
}

func (pb *progressBar) Step() {
	pb.Lock()
	defer pb.Unlock()

	pb.current++
	pb.Update()
}

func (pb *progressBar) Interrupt(callback func()) {
	pb.Lock()
	defer pb.Unlock()
	pb.Stop()
	defer pb.Update()

	callback()
}

// --- Parallelization ---------------------------------------------------------

type parallelFunc func(item string) (message string, err error)

// parallelize runs the callback in parallel on a list of strings,
// reporting progress and any non-fatal messages.
func parallelize(strings []string, callback parallelFunc) error {
	pb := newProgressBar(len(strings))
	defer pb.Stop()

	ctx, cancel := context.WithCancelCause(context.Background())
	wg := sync.WaitGroup{}
	for _, item := range strings {
		if taskSemaphore.acquire(ctx) != nil {
			break
		}

		wg.Add(1)
		go func(item string) {
			defer taskSemaphore.release()
			defer wg.Done()
			if message, err := callback(item); err != nil {
				cancel(err)
			} else if message != "" {
				pb.Interrupt(func() { log.Printf("%s: %s\n", item, message) })
			}
			pb.Step()
		}(item)
	}
	wg.Wait()
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}
	return nil
}

// --- Initialization ----------------------------------------------------------

// cmdInit initializes a "gallery directory" that contains gallery.sqlite,
// images, thumbs.
func cmdInit(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return errWrongUsage
	}
	if err := openDB(fs.Arg(0)); err != nil {
		return err
	}

	if _, err := db.Exec(initializeSQL); err != nil {
		return err
	}

	// XXX: There's technically no reason to keep images as symlinks,
	// we might just keep absolute paths in the database as well.
	if err := os.MkdirAll(
		filepath.Join(galleryDirectory, nameOfImageRoot), 0755); err != nil {
		return err
	}
	if err := os.MkdirAll(
		filepath.Join(galleryDirectory, nameOfThumbRoot), 0755); err != nil {
		return err
	}
	return nil
}

// --- API: Browse -------------------------------------------------------------

func getSubdirectories(tx *sql.Tx, parent int64) (names []string, err error) {
	return dbCollectStrings(`SELECT name FROM node
		WHERE IFNULL(parent, 0) = ? AND sha1 IS NULL`, parent)
}

type webEntry struct {
	SHA1     string `json:"sha1"`
	Name     string `json:"name"`
	Modified int64  `json:"modified"`
	ThumbW   int64  `json:"thumbW"`
	ThumbH   int64  `json:"thumbH"`
}

func getSubentries(tx *sql.Tx, parent int64) (entries []webEntry, err error) {
	rows, err := tx.Query(`
		SELECT i.sha1, n.name, n.mtime, IFNULL(i.thumbw, 0), IFNULL(i.thumbh, 0)
		FROM node AS n
		JOIN image AS i ON n.sha1 = i.sha1
		WHERE n.parent = ?`, parent)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	entries = []webEntry{}
	for rows.Next() {
		var e webEntry
		if err = rows.Scan(
			&e.SHA1, &e.Name, &e.Modified, &e.ThumbW, &e.ThumbH); err != nil {
			return nil, err
		}
		entries = append(entries, e)
	}
	return entries, rows.Err()
}

func handleAPIBrowse(w http.ResponseWriter, r *http.Request) {
	var params struct {
		Path string
	}
	if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var result struct {
		Subdirectories []string   `json:"subdirectories"`
		Entries        []webEntry `json:"entries"`
	}

	tx, err := db.Begin()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer tx.Rollback()

	parent, err := idForDirectoryPath(tx, decodeWebPath(params.Path), false)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	result.Subdirectories, err = getSubdirectories(tx, parent)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	result.Entries, err = getSubentries(tx, parent)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := json.NewEncoder(w).Encode(result); err != nil {
		log.Println(err)
	}
}

// --- API: Tags ---------------------------------------------------------------

type webTagNamespace struct {
	Description string           `json:"description"`
	Tags        map[string]int64 `json:"tags"`
}

func getTags(nsID int64) (result map[string]int64, err error) {
	rows, err := db.Query(`
		SELECT t.name, COUNT(ta.tag) AS count
		FROM tag AS t
		LEFT JOIN tag_assignment AS ta ON t.id = ta.tag
		WHERE t.space = ?
		GROUP BY t.id`, nsID)
	if err != nil {
		return
	}
	defer rows.Close()

	result = make(map[string]int64)
	for rows.Next() {
		var (
			name  string
			count int64
		)
		if err = rows.Scan(&name, &count); err != nil {
			return
		}
		result[name] = count
	}
	return result, rows.Err()
}

func getTagNamespaces(match *string) (
	result map[string]webTagNamespace, err error) {
	var rows *sql.Rows
	if match != nil {
		rows, err = db.Query(`SELECT id, name, IFNULL(description, '')
			FROM tag_space WHERE name = ?`, *match)
	} else {
		rows, err = db.Query(`SELECT id, name, IFNULL(description, '')
			FROM tag_space`)
	}
	if err != nil {
		return
	}
	defer rows.Close()

	result = make(map[string]webTagNamespace)
	for rows.Next() {
		var (
			id   int64
			name string
			ns   webTagNamespace
		)
		if err = rows.Scan(&id, &name, &ns.Description); err != nil {
			return
		}
		if ns.Tags, err = getTags(id); err != nil {
			return
		}
		result[name] = ns
	}
	return result, rows.Err()
}

func handleAPITags(w http.ResponseWriter, r *http.Request) {
	var params struct {
		Namespace *string
	}
	if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result, err := getTagNamespaces(params.Namespace)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if err := json.NewEncoder(w).Encode(result); err != nil {
		log.Println(err)
	}
}

// --- API: Duplicates ---------------------------------------------------------

type webDuplicateImage struct {
	SHA1       string `json:"sha1"`
	ThumbW     int64  `json:"thumbW"`
	ThumbH     int64  `json:"thumbH"`
	Occurences int64  `json:"occurences"`
}

// A hamming distance of zero (direct dhash match) will be more than sufficient.
const duplicatesCTE = `WITH
	duplicated(dhash, count) AS (
		SELECT dhash, COUNT(*) AS count FROM image
		WHERE dhash IS NOT NULL
		GROUP BY dhash HAVING count > 1
	),
	multipathed(sha1, count) AS (
		SELECT n.sha1, COUNT(*) AS count FROM node AS n
		JOIN image AS i ON i.sha1 = n.sha1
		WHERE i.dhash IS NULL
		OR i.dhash NOT IN (SELECT dhash FROM duplicated)
		GROUP BY n.sha1 HAVING count > 1
	)
`

func getDuplicatesSimilar(stmt *sql.Stmt, dhash int64) (
	result []webDuplicateImage, err error) {
	rows, err := stmt.Query(dhash)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result = []webDuplicateImage{}
	for rows.Next() {
		var image webDuplicateImage
		if err = rows.Scan(&image.SHA1, &image.ThumbW, &image.ThumbH,
			&image.Occurences); err != nil {
			return nil, err
		}
		result = append(result, image)
	}
	return result, rows.Err()
}

func getDuplicates1(result [][]webDuplicateImage) (
	[][]webDuplicateImage, error) {
	stmt, err := db.Prepare(`
		SELECT i.sha1, IFNULL(i.thumbw, 0), IFNULL(i.thumbh, 0),
			COUNT(*) AS occurences
		FROM image AS i
		JOIN node AS n ON n.sha1 = i.sha1
		WHERE i.dhash = ?
		GROUP BY n.sha1`)
	if err != nil {
		return nil, err
	}
	defer stmt.Close()

	rows, err := db.Query(duplicatesCTE + `SELECT dhash FROM duplicated`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var (
			group []webDuplicateImage
			dhash int64
		)
		if err = rows.Scan(&dhash); err != nil {
			return nil, err
		}
		if group, err = getDuplicatesSimilar(stmt, dhash); err != nil {
			return nil, err
		}
		result = append(result, group)
	}
	return result, rows.Err()
}

func getDuplicates2(result [][]webDuplicateImage) (
	[][]webDuplicateImage, error) {
	stmt, err := db.Prepare(`
		SELECT i.sha1, IFNULL(i.thumbw, 0), IFNULL(i.thumbh, 0),
			COUNT(*) AS occurences
		FROM image AS i
		JOIN node AS n ON n.sha1 = i.sha1
		WHERE i.sha1 = ?
		GROUP BY n.sha1`)
	if err != nil {
		return nil, err
	}
	defer stmt.Close()

	rows, err := db.Query(duplicatesCTE + `SELECT sha1 FROM multipathed`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var (
			image webDuplicateImage
			sha1  string
		)
		if err = rows.Scan(&sha1); err != nil {
			return nil, err
		}
		if err := stmt.QueryRow(sha1).Scan(&image.SHA1,
			&image.ThumbW, &image.ThumbH, &image.Occurences); err != nil {
			return nil, err
		}
		result = append(result, []webDuplicateImage{image})
	}
	return result, rows.Err()
}

func handleAPIDuplicates(w http.ResponseWriter, r *http.Request) {
	var params struct{}
	if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var (
		result = [][]webDuplicateImage{}
		err    error
	)
	if result, err = getDuplicates1(result); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if result, err = getDuplicates2(result); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if err := json.NewEncoder(w).Encode(result); err != nil {
		log.Println(err)
	}
}

// --- API: Orphans ------------------------------------------------------------

type webOrphanImage struct {
	SHA1   string `json:"sha1"`
	ThumbW int64  `json:"thumbW"`
	ThumbH int64  `json:"thumbH"`
	Tags   int64  `json:"tags"`
}

type webOrphan struct {
	webOrphanImage
	LastPath    string          `json:"lastPath"`
	Replacement *webOrphanImage `json:"replacement"`
}

func getOrphanReplacement(webPath string) (*webOrphanImage, error) {
	tx, err := db.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	path := decodeWebPath(webPath)
	if len(path) == 0 {
		return nil, nil
	}

	parent, err := idForDirectoryPath(tx, path[:len(path)-1], false)
	if err != nil {
		return nil, err
	}

	var image webOrphanImage
	err = db.QueryRow(`SELECT i.sha1,
		IFNULL(i.thumbw, 0), IFNULL(i.thumbh, 0), COUNT(ta.sha1) AS tags
		FROM node AS n
		JOIN image AS i ON n.sha1 = i.sha1
		LEFT JOIN tag_assignment AS ta ON n.sha1 = ta.sha1
		WHERE n.parent = ? AND n.name = ?
		GROUP BY n.sha1`, parent, path[len(path)-1]).Scan(
		&image.SHA1, &image.ThumbW, &image.ThumbH, &image.Tags)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	return &image, nil
}

func getOrphans() (result []webOrphan, err error) {
	rows, err := db.Query(`SELECT o.sha1, o.path,
		IFNULL(i.thumbw, 0), IFNULL(i.thumbh, 0), COUNT(ta.sha1) AS tags
		FROM orphan AS o
		JOIN image AS i ON o.sha1 = i.sha1
		LEFT JOIN tag_assignment AS ta ON o.sha1 = ta.sha1
		GROUP BY o.sha1`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result = []webOrphan{}
	for rows.Next() {
		var orphan webOrphan
		if err = rows.Scan(&orphan.SHA1, &orphan.LastPath,
			&orphan.ThumbW, &orphan.ThumbH, &orphan.Tags); err != nil {
			return nil, err
		}

		orphan.Replacement, err = getOrphanReplacement(orphan.LastPath)
		if err != nil {
			return nil, err
		}

		result = append(result, orphan)
	}
	return result, rows.Err()
}

func handleAPIOrphans(w http.ResponseWriter, r *http.Request) {
	var params struct{}
	if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result, err := getOrphans()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if err := json.NewEncoder(w).Encode(result); err != nil {
		log.Println(err)
	}
}

// --- API: Image view ---------------------------------------------------------

func getImageDimensions(sha1 string) (w int64, h int64, err error) {
	err = db.QueryRow(`SELECT width, height FROM image WHERE sha1 = ?`,
		sha1).Scan(&w, &h)
	return
}

func getImagePaths(sha1 string) (paths []string, err error) {
	rows, err := db.Query(`WITH RECURSIVE paths(parent, path) AS (
		SELECT parent, name AS path FROM node WHERE sha1 = ?
		UNION ALL
		SELECT n.parent, n.name || '/' || p.path
		FROM node AS n JOIN paths AS p ON n.id = p.parent
	) SELECT path FROM paths WHERE parent IS NULL`, sha1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	paths = []string{}
	for rows.Next() {
		var path string
		if err := rows.Scan(&path); err != nil {
			return nil, err
		}
		paths = append(paths, path)
	}
	return paths, rows.Err()
}

func getImageTags(sha1 string) (map[string]map[string]float32, error) {
	rows, err := db.Query(`
		SELECT ts.name, t.name, ta.weight FROM tag_assignment AS ta
		JOIN tag AS t ON t.id = ta.tag
		JOIN tag_space AS ts ON ts.id = t.space
		WHERE ta.sha1 = ?`, sha1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := make(map[string]map[string]float32)
	for rows.Next() {
		var (
			space, tag string
			weight     float32
		)
		if err := rows.Scan(&space, &tag, &weight); err != nil {
			return nil, err
		}

		tags := result[space]
		if tags == nil {
			tags = make(map[string]float32)
			result[space] = tags
		}
		tags[tag] = weight
	}
	return result, rows.Err()
}

func handleAPIInfo(w http.ResponseWriter, r *http.Request) {
	var params struct {
		SHA1 string
	}
	if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var result struct {
		Width  int64                         `json:"width"`
		Height int64                         `json:"height"`
		Paths  []string                      `json:"paths"`
		Tags   map[string]map[string]float32 `json:"tags"`
	}

	var err error
	result.Width, result.Height, err = getImageDimensions(params.SHA1)
	if errors.Is(err, sql.ErrNoRows) {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	} else if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	result.Paths, err = getImagePaths(params.SHA1)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	result.Tags, err = getImageTags(params.SHA1)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := json.NewEncoder(w).Encode(result); err != nil {
		log.Println(err)
	}
}

// --- API: Image similar ------------------------------------------------------

type webSimilarImage struct {
	SHA1        string   `json:"sha1"`
	PixelsRatio float32  `json:"pixelsRatio"`
	ThumbW      int64    `json:"thumbW"`
	ThumbH      int64    `json:"thumbH"`
	Paths       []string `json:"paths"`
}

func getSimilar(sha1 string, dhash int64, pixels int64, distance int) (
	result []webSimilarImage, err error) {
	// For distance ∈ {0, 1}, this query is quite inefficient.
	// In exchange, it's generic.
	//
	// If there's a dhash, there should also be thumbnail dimensions,
	// so not bothering with IFNULL on them.
	rows, err := db.Query(`
		SELECT sha1, width * height, IFNULL(thumbw, 0), IFNULL(thumbh, 0)
		FROM image WHERE sha1 <> ? AND dhash IS NOT NULL
		AND hamming(dhash, ?) = ?`, sha1, dhash, distance)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result = []webSimilarImage{}
	for rows.Next() {
		var (
			match       webSimilarImage
			matchPixels int64
		)
		if err = rows.Scan(&match.SHA1,
			&matchPixels, &match.ThumbW, &match.ThumbH); err != nil {
			return nil, err
		}
		if match.Paths, err = getImagePaths(match.SHA1); err != nil {
			return nil, err
		}
		match.PixelsRatio = float32(matchPixels) / float32(pixels)
		result = append(result, match)
	}
	return result, rows.Err()
}

func getSimilarGroups(sha1 string, dhash int64, pixels int64,
	output map[string][]webSimilarImage) error {
	var err error
	for distance := 0; distance <= 1; distance++ {
		output[fmt.Sprintf("Perceptual distance %d", distance)], err =
			getSimilar(sha1, dhash, pixels, distance)
		if err != nil {
			return err
		}
	}
	return nil
}

func handleAPISimilar(w http.ResponseWriter, r *http.Request) {
	var params struct {
		SHA1 string
	}
	if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var result struct {
		Info   webSimilarImage              `json:"info"`
		Groups map[string][]webSimilarImage `json:"groups"`
	}

	result.Info = webSimilarImage{SHA1: params.SHA1, PixelsRatio: 1}
	if paths, err := getImagePaths(params.SHA1); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	} else {
		result.Info.Paths = paths
	}

	var (
		width, height int64
		dhash         sql.NullInt64
	)
	err := db.QueryRow(`
		SELECT width, height, dhash, IFNULL(thumbw, 0), IFNULL(thumbh, 0)
		FROM image WHERE sha1 = ?`, params.SHA1).Scan(&width, &height, &dhash,
		&result.Info.ThumbW, &result.Info.ThumbH)
	if errors.Is(err, sql.ErrNoRows) {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	} else if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	result.Groups = make(map[string][]webSimilarImage)
	if dhash.Valid {
		if err := getSimilarGroups(
			params.SHA1, dhash.Int64, width*height, result.Groups); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	if err := json.NewEncoder(w).Encode(result); err != nil {
		log.Println(err)
	}
}

// --- API: Search -------------------------------------------------------------
// The SQL building is the most miserable part of the whole program.

const searchCTE1 = `WITH
	matches(sha1, thumbw, thumbh, score) AS (
		SELECT i.sha1, i.thumbw, i.thumbh, ta.weight AS score
		FROM tag_assignment AS ta
		JOIN image AS i ON i.sha1 = ta.sha1
		WHERE ta.tag = %d
	)
`

const searchCTEMulti = `WITH
	positive(tag) AS (VALUES %s),
	filtered(sha1) AS (%s),
	matches(sha1, thumbw, thumbh, score) AS (
		SELECT i.sha1, i.thumbw, i.thumbh,
			product(IFNULL(ta.weight, 0)) AS score
		FROM image AS i, positive AS p
		JOIN filtered AS c ON i.sha1 = c.sha1
		LEFT JOIN tag_assignment AS ta ON ta.sha1 = i.sha1 AND ta.tag = p.tag
		GROUP BY i.sha1
	)
`

func searchQueryToCTE(tx *sql.Tx, query string) (string, error) {
	positive, negative := []int64{}, []int64{}
	for _, word := range strings.Split(query, " ") {
		if word == "" {
			continue
		}

		space, tag, _ := strings.Cut(word, ":")

		negated := false
		if strings.HasPrefix(space, "-") {
			space = space[1:]
			negated = true
		}

		var tagID int64
		err := tx.QueryRow(`
			SELECT t.id FROM tag AS t
			JOIN tag_space AS ts ON t.space = ts.id
			WHERE ts.name = ? AND t.name = ?`, space, tag).Scan(&tagID)
		if err != nil {
			return "", err
		}

		if negated {
			negative = append(negative, tagID)
		} else {
			positive = append(positive, tagID)
		}
	}

	// Don't return most of the database, and simplify the following builder.
	if len(positive) == 0 {
		return "", errors.New("search is too wide")
	}

	// Optimise single tag searches.
	if len(positive) == 1 && len(negative) == 0 {
		return fmt.Sprintf(searchCTE1, positive[0]), nil
	}

	values := fmt.Sprintf(`(%d)`, positive[0])
	filtered := fmt.Sprintf(
		`SELECT sha1 FROM tag_assignment WHERE tag = %d`, positive[0])
	for _, tagID := range positive[1:] {
		values += fmt.Sprintf(`, (%d)`, tagID)
		filtered += fmt.Sprintf(` INTERSECT
			SELECT sha1 FROM tag_assignment WHERE tag = %d`, tagID)
	}
	for _, tagID := range negative {
		filtered += fmt.Sprintf(` EXCEPT
			SELECT sha1 FROM tag_assignment WHERE tag = %d`, tagID)
	}

	return fmt.Sprintf(searchCTEMulti, values, filtered), nil
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

type webTagMatch struct {
	SHA1   string  `json:"sha1"`
	ThumbW int64   `json:"thumbW"`
	ThumbH int64   `json:"thumbH"`
	Score  float32 `json:"score"`
}

func getTagMatches(tx *sql.Tx, cte string) (matches []webTagMatch, err error) {
	rows, err := tx.Query(cte + `
		SELECT sha1, IFNULL(thumbw, 0), IFNULL(thumbh, 0), score
		FROM matches`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	matches = []webTagMatch{}
	for rows.Next() {
		var match webTagMatch
		if err = rows.Scan(&match.SHA1,
			&match.ThumbW, &match.ThumbH, &match.Score); err != nil {
			return nil, err
		}
		matches = append(matches, match)
	}
	return matches, rows.Err()
}

type webTagSupertag struct {
	space string
	tag   string
	score float32
}

func getTagSupertags(tx *sql.Tx, cte string) (
	result map[int64]*webTagSupertag, err error) {
	rows, err := tx.Query(cte + `
		SELECT DISTINCT ta.tag, ts.name, t.name
		FROM tag_assignment AS ta
		JOIN matches AS m ON m.sha1 = ta.sha1
		JOIN tag AS t ON ta.tag = t.id
		JOIN tag_space AS ts ON ts.id = t.space`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result = make(map[int64]*webTagSupertag)
	for rows.Next() {
		var (
			tag int64
			st  webTagSupertag
		)
		if err = rows.Scan(&tag, &st.space, &st.tag); err != nil {
			return nil, err
		}
		result[tag] = &st
	}
	return result, rows.Err()
}

type webTagRelated struct {
	Tag   string  `json:"tag"`
	Score float32 `json:"score"`
}

func getTagRelated(tx *sql.Tx, cte string, matches int) (
	result map[string][]webTagRelated, err error) {
	// Not sure if this level of efficiency is achievable directly in SQL.
	supertags, err := getTagSupertags(tx, cte)
	if err != nil {
		return nil, err
	}

	rows, err := tx.Query(cte + `
		SELECT ta.tag, ta.weight
		FROM tag_assignment AS ta
		JOIN matches AS m ON m.sha1 = ta.sha1`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var (
			tag    int64
			weight float32
		)
		if err = rows.Scan(&tag, &weight); err != nil {
			return nil, err
		}
		supertags[tag].score += weight
	}

	result = make(map[string][]webTagRelated)
	for _, info := range supertags {
		if score := info.score / float32(matches); score >= 0.1 {
			r := webTagRelated{Tag: info.tag, Score: score}
			result[info.space] = append(result[info.space], r)
		}
	}
	return result, rows.Err()
}

func handleAPISearch(w http.ResponseWriter, r *http.Request) {
	var params struct {
		Query string
	}
	if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var result struct {
		Matches []webTagMatch              `json:"matches"`
		Related map[string][]webTagRelated `json:"related"`
	}

	tx, err := db.Begin()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer tx.Rollback()

	cte, err := searchQueryToCTE(tx, params.Query)
	if errors.Is(err, sql.ErrNoRows) {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	} else if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if result.Matches, err = getTagMatches(tx, cte); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if result.Related, err = getTagRelated(tx, cte,
		len(result.Matches)); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if err := json.NewEncoder(w).Encode(result); err != nil {
		log.Println(err)
	}
}

// --- Web ---------------------------------------------------------------------

var hashRE = regexp.MustCompile(`^/.*?/([0-9a-f]{40})$`)
var staticHandler http.Handler

var page = template.Must(template.New("/").Parse(`<!DOCTYPE html><html><head>
	<title>Gallery</title>
	<meta charset="utf-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<link rel=stylesheet href=style.css>
</head><body>
	<noscript>This is a web application, and requires Javascript.</noscript>
	<script src=mithril.js></script>
	<script src=gallery.js></script>
</body></html>`))

func handleRequest(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		staticHandler.ServeHTTP(w, r)
		return
	}
	if err := page.Execute(w, nil); err != nil {
		log.Println(err)
	}
}

func handleImages(w http.ResponseWriter, r *http.Request) {
	if m := hashRE.FindStringSubmatch(r.URL.Path); m == nil {
		http.NotFound(w, r)
	} else {
		http.ServeFile(w, r, imagePath(m[1]))
	}
}

func handleThumbs(w http.ResponseWriter, r *http.Request) {
	if m := hashRE.FindStringSubmatch(r.URL.Path); m == nil {
		http.NotFound(w, r)
	} else {
		http.ServeFile(w, r, thumbPath(m[1]))
	}
}

// cmdWeb runs a web UI against GD on ADDRESS.
func cmdWeb(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return errWrongUsage
	}
	if err := openDB(fs.Arg(0)); err != nil {
		return err
	}

	address := fs.Arg(1)

	// This separation is not strictly necessary,
	// but having an elementary level of security doesn't hurt either.
	staticHandler = http.FileServer(http.Dir("public"))

	http.HandleFunc("/", handleRequest)
	http.HandleFunc("/image/", handleImages)
	http.HandleFunc("/thumb/", handleThumbs)
	http.HandleFunc("/api/browse", handleAPIBrowse)
	http.HandleFunc("/api/tags", handleAPITags)
	http.HandleFunc("/api/duplicates", handleAPIDuplicates)
	http.HandleFunc("/api/orphans", handleAPIOrphans)
	http.HandleFunc("/api/info", handleAPIInfo)
	http.HandleFunc("/api/similar", handleAPISimilar)
	http.HandleFunc("/api/search", handleAPISearch)

	host, port, err := net.SplitHostPort(address)
	if err != nil {
		log.Println(err)
	} else if host == "" {
		log.Println("http://" + net.JoinHostPort("localhost", port))
	} else {
		log.Println("http://" + address)
	}

	s := &http.Server{
		Addr:           address,
		ReadTimeout:    60 * time.Second,
		WriteTimeout:   60 * time.Second,
		MaxHeaderBytes: 32 << 10,
	}
	return s.ListenAndServe()
}

// --- Sync --------------------------------------------------------------------

type syncFileInfo struct {
	dbID     int64  // DB node ID, or zero if there was none
	dbParent int64  // where the file was to be stored
	dbName   string // the name under which it was to be stored
	fsPath   string // symlink target
	fsMtime  int64  // last modified Unix timestamp, used a bit like an ID

	err    error  // any processing error
	sha1   string // raw content hash, empty to skip file
	width  int    // image width in pixels
	height int    // image height in pixels
}

type syncContext struct {
	ctx  context.Context
	tx   *sql.Tx
	info chan syncFileInfo
	pb   *progressBar

	stmtOrphan     *sql.Stmt
	stmtDisposeSub *sql.Stmt
	stmtDisposeAll *sql.Stmt

	// linked tracks which image hashes we've checked so far in the run.
	linked map[string]struct{}
}

func syncPrintf(c *syncContext, format string, v ...any) {
	c.pb.Interrupt(func() { log.Printf(format+"\n", v...) })
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

type syncNode struct {
	dbID    int64
	dbName  string
	dbMtime int64
	dbSHA1  string
}

func (n *syncNode) dbIsDir() bool { return n.dbSHA1 == "" }

type syncFile struct {
	fsName  string
	fsMtime int64
	fsIsDir bool
}

type syncPair struct {
	db *syncNode
	fs *syncFile
}

// syncGetNodes returns direct children of a DB node, ordered by name.
// SQLite, like Go, compares strings byte-wise by default.
func syncGetNodes(tx *sql.Tx, dbParent int64) (nodes []syncNode, err error) {
	// This works even for the root, which doesn't exist as a DB node.
	rows, err := tx.Query(`SELECT id, name, IFNULL(mtime, 0), IFNULL(sha1, '')
		FROM node WHERE IFNULL(parent, 0) = ? ORDER BY name`, dbParent)
	if err != nil {
		return
	}
	defer rows.Close()

	for rows.Next() {
		var node syncNode
		if err = rows.Scan(&node.dbID,
			&node.dbName, &node.dbMtime, &node.dbSHA1); err != nil {
			return
		}
		nodes = append(nodes, node)
	}
	return nodes, rows.Err()
}

// syncGetFiles returns direct children of a FS directory, ordered by name.
func syncGetFiles(fsPath string) (files []syncFile, err error) {
	dir, err := os.Open(fsPath)
	if err != nil {
		return
	}
	defer dir.Close()

	entries, err := dir.ReadDir(0)
	if err != nil {
		return
	}

	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			return files, err
		}

		files = append(files, syncFile{
			fsName:  entry.Name(),
			fsMtime: info.ModTime().Unix(),
			fsIsDir: entry.IsDir(),
		})
	}
	sort.Slice(files,
		func(a, b int) bool { return files[a].fsName < files[b].fsName })
	return
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

func syncIsImage(path string) (bool, error) {
	out, err := exec.Command("xdg-mime", "query", "filetype", path).Output()
	if err != nil {
		return false, err
	}

	return bytes.HasPrefix(out, []byte("image/")), nil
}

func syncPingImage(path string) (int, int, error) {
	out, err := exec.Command("identify", "-limit", "thread", "1",
		"-ping", "-format", "%w %h", path+"[0]").Output()
	if err != nil {
		return 0, 0, err
	}

	var w, h int
	_, err = fmt.Fscanf(bytes.NewReader(out), "%d %d", &w, &h)
	return w, h, err
}

func syncProcess(c *syncContext, info *syncFileInfo) error {
	// Skip videos, which ImageMagick can process, but we don't want it to,
	// so that they're not converted 1:1 to WebP.
	pathIsImage, err := syncIsImage(info.fsPath)
	if err != nil {
		return err
	}
	if !pathIsImage {
		return nil
	}

	info.width, info.height, err = syncPingImage(info.fsPath)
	if err != nil {
		return err
	}

	f, err := os.Open(info.fsPath)
	if err != nil {
		return err
	}
	defer f.Close()

	// We could make this at least somewhat interruptible by c.ctx,
	// though it would still work poorly.
	hash := sha1.New()
	_, err = io.CopyBuffer(hash, f, make([]byte, 65536))
	if err != nil {
		return err
	}

	info.sha1 = hex.EncodeToString(hash.Sum(nil))
	return nil
}

// syncEnqueue runs file scanning, which can be CPU and I/O expensive,
// in parallel. The goroutine only touches the filesystem, read-only.
func syncEnqueue(c *syncContext, info syncFileInfo) error {
	if err := taskSemaphore.acquire(c.ctx); err != nil {
		return err
	}

	go func(info syncFileInfo) {
		defer taskSemaphore.release()
		info.err = syncProcess(c, &info)
		c.info <- info
	}(info)
	return nil
}

// syncDequeue flushes the result queue of finished asynchronous tasks.
func syncDequeue(c *syncContext) error {
	for {
		select {
		case <-c.ctx.Done():
			return c.ctx.Err()
		case info := <-c.info:
			if err := syncPostProcess(c, info); err != nil {
				return err
			}
		default:
			return nil
		}
	}
}

// syncDispose creates orphan records for the entire subtree given by nodeID
// as appropriate, then deletes all nodes within the subtree. The subtree root
// node is not deleted if "keepNode" is true.
//
// Orphans keep their thumbnail files, as evidence.
func syncDispose(c *syncContext, nodeID int64, keepNode bool) error {
	if _, err := c.stmtOrphan.Exec(nodeID); err != nil {
		return err
	}

	if keepNode {
		if _, err := c.stmtDisposeSub.Exec(nodeID); err != nil {
			return err
		}
	} else {
		if _, err := c.stmtDisposeAll.Exec(nodeID); err != nil {
			return err
		}
	}
	return nil
}

func syncImageResave(c *syncContext, path string, target string) error {
	dirname, _ := filepath.Split(path)
	if err := os.MkdirAll(dirname, 0755); err != nil {
		return err
	}

	for {
		// Try to remove anything standing in the way.
		err := os.Remove(path)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}

		// TODO: Make it possible to copy or reflink (ioctl FICLONE).
		err = os.Symlink(target, path)
		if err == nil || !errors.Is(err, fs.ErrExist) {
			return err
		}
	}
}

func syncImageSave(c *syncContext, sha1 string, target string) error {
	if _, ok := c.linked[sha1]; ok {
		return nil
	}

	ok, path := false, imagePath(sha1)
	if link, err := os.Readlink(path); err == nil {
		ok = link == target
	} else {
		// If it exists, but it is not a symlink, let it be.
		// Even though it may not be a regular file.
		ok = errors.Is(err, syscall.EINVAL)
	}

	if !ok {
		if err := syncImageResave(c, path, target); err != nil {
			return err
		}
	}

	c.linked[sha1] = struct{}{}
	return nil
}

func syncImage(c *syncContext, info syncFileInfo) error {
	if _, err := c.tx.Exec(`INSERT INTO image(sha1, width, height)
		VALUES (?, ?, ?) ON CONFLICT(sha1) DO NOTHING`,
		info.sha1, info.width, info.height); err != nil {
		return err
	}

	return syncImageSave(c, info.sha1, info.fsPath)
}

func syncPostProcess(c *syncContext, info syncFileInfo) error {
	defer c.pb.Step()

	// TODO: When replacing an image node (whether it has or doesn't have
	// other links to keep it alive), we could offer copying all tags,
	// though this needs another table to track it.
	// (If it's equivalent enough, the dhash will stay the same,
	// so user can resolve this through the duplicates feature.)
	switch {
	case info.err != nil:
		// * → error
		if ee, ok := info.err.(*exec.ExitError); ok {
			message := string(ee.Stderr)
			if message == "" {
				message = ee.String()
			}
			syncPrintf(c, "%s: %s", info.fsPath, message)
		} else {
			return info.err
		}
		fallthrough

	case info.sha1 == "":
		// 0 → 0
		if info.dbID == 0 {
			return nil
		}

		// D → 0, F → 0
		// TODO: Make it possible to disable removal (for copying only?)
		return syncDispose(c, info.dbID, false /*keepNode*/)

	case info.dbID == 0:
		// 0 → F
		if err := syncImage(c, info); err != nil {
			return err
		}
		if _, err := c.tx.Exec(`INSERT INTO node(parent, name, mtime, sha1)
			VALUES (?, ?, ?, ?)`,
			info.dbParent, info.dbName, info.fsMtime, info.sha1); err != nil {
			return err
		}
		return nil

	default:
		// D → F, F → F (this statement is a no-op with the latter)
		if err := syncDispose(c, info.dbID, true /*keepNode*/); err != nil {
			return err
		}

		// Even if the hash didn't change, see comment in syncDirectoryPair().
		if err := syncImage(c, info); err != nil {
			return err
		}
		if _, err := c.tx.Exec(`UPDATE node SET mtime = ?, sha1 = ?
			WHERE id = ?`, info.fsMtime, info.sha1, info.dbID); err != nil {
			return err
		}
		return nil
	}
}

func syncDirectoryPair(c *syncContext, dbParent int64, fsPath string,
	pair syncPair) error {
	db, fs, fsInfo := pair.db, pair.fs, syncFileInfo{dbParent: dbParent}
	if db != nil {
		fsInfo.dbID = db.dbID
	}
	if fs != nil {
		fsInfo.dbName = fs.fsName
		fsInfo.fsPath = filepath.Join(fsPath, fs.fsName)
		fsInfo.fsMtime = fs.fsMtime
	}

	switch {
	case db == nil && fs == nil:
		// 0 → 0, unreachable.

	case db == nil && fs.fsIsDir:
		// 0 → D
		var id int64
		if result, err := c.tx.Exec(`INSERT INTO node(parent, name)
			VALUES (?, ?)`, dbParent, fs.fsName); err != nil {
			return err
		} else if id, err = result.LastInsertId(); err != nil {
			return err
		}
		return syncDirectory(c, id, fsInfo.fsPath)

	case db == nil:
		// 0 → F (or 0 → 0)
		return syncEnqueue(c, fsInfo)

	case fs == nil:
		// D → 0, F → 0
		// TODO: Make it possible to disable removal (for copying only?)
		return syncDispose(c, db.dbID, false /*keepNode*/)

	case db.dbIsDir() && fs.fsIsDir:
		// D → D
		return syncDirectory(c, db.dbID, fsInfo.fsPath)

	case db.dbIsDir():
		// D → F (or D → 0)
		return syncEnqueue(c, fsInfo)

	case fs.fsIsDir:
		// F → D
		if err := syncDispose(c, db.dbID, true /*keepNode*/); err != nil {
			return err
		}
		if _, err := c.tx.Exec(`UPDATE node
			SET mtime = NULL, sha1 = NULL WHERE id = ?`, db.dbID); err != nil {
			return err
		}
		return syncDirectory(c, db.dbID, fsInfo.fsPath)

	case db.dbMtime != fs.fsMtime:
		// F → F (or F → 0)
		// Assuming that any content modifications change the timestamp.
		return syncEnqueue(c, fsInfo)

	default:
		// F → F
		// Try to fix symlinks, to handle the following situations:
		//  1. Image A occurs in paths 1 and 2, we use a symlink to path 1,
		//     and path 1 is removed from the filesystem:
		//     path 2 would not resolve if the mtime didn't change.
		//  2. Image A occurs in paths 1 and 2, we use a symlink to path 1,
		//     and path 1 is changed:
		//     path 2 would resolve to the wrong file.
		// This may relink images with multiple occurences unnecessarily,
		// but it will always fix the roots that are being synced.
		if err := syncImageSave(c, db.dbSHA1, fsInfo.fsPath); err != nil {
			return err
		}
	}
	return nil
}

func syncDirectory(c *syncContext, dbParent int64, fsPath string) error {
	db, err := syncGetNodes(c.tx, dbParent)
	if err != nil {
		return err
	}

	fs, err := syncGetFiles(fsPath)
	if err != nil {
		return err
	}

	// This would not be fatal, but it has annoying consequences.
	if _, ok := slices.BinarySearchFunc(fs, syncFile{fsName: nameOfDB},
		func(a, b syncFile) int {
			return strings.Compare(a.fsName, b.fsName)
		}); ok {
		syncPrintf(c, "%s may be a gallery directory, treating as empty",
			fsPath)
		fs = nil
	}

	// Convert differences to a form more convenient for processing.
	iDB, iFS, pairs := 0, 0, []syncPair{}
	for iDB < len(db) && iFS < len(fs) {
		if db[iDB].dbName == fs[iFS].fsName {
			pairs = append(pairs, syncPair{&db[iDB], &fs[iFS]})
			iDB++
			iFS++
		} else if db[iDB].dbName < fs[iFS].fsName {
			pairs = append(pairs, syncPair{&db[iDB], nil})
			iDB++
		} else {
			pairs = append(pairs, syncPair{nil, &fs[iFS]})
			iFS++
		}
	}
	for i := range db[iDB:] {
		pairs = append(pairs, syncPair{&db[iDB+i], nil})
	}
	for i := range fs[iFS:] {
		pairs = append(pairs, syncPair{nil, &fs[iFS+i]})
	}

	for _, pair := range pairs {
		if err := syncDequeue(c); err != nil {
			return err
		}
		if err := syncDirectoryPair(c, dbParent, fsPath, pair); err != nil {
			return err
		}
	}
	return nil
}

func syncRoot(c *syncContext, dbPath []string, fsPath string) error {
	// TODO: Support synchronizing individual files.
	// This can only be treated as 0 → F, F → F, or D → F, that is,
	// a variation on current syncEnqueue(), but dbParent must be nullable.

	// Figure out a database root (not trying to convert F → D on conflict,
	// also because we don't know yet if the argument is a directory).
	//
	// Synchronizing F → D or * → F are special cases not worth implementing.
	dbParent, err := idForDirectoryPath(c.tx, dbPath, true)
	if err != nil {
		return err
	}
	if err := syncDirectory(c, dbParent, fsPath); err != nil {
		return err
	}

	// Wait for all tasks to finish, and process the results of their work.
	for i := 0; i < cap(taskSemaphore); i++ {
		if err := taskSemaphore.acquire(c.ctx); err != nil {
			return err
		}
	}
	if err := syncDequeue(c); err != nil {
		return err
	}

	// This is not our semaphore, so prepare it for the next user.
	for i := 0; i < cap(taskSemaphore); i++ {
		taskSemaphore.release()
	}

	// Delete empty directories, from the bottom of the tree up to,
	// but not including, the inserted root.
	//
	// We need to do this at the end due to our recursive handling,
	// as well as because of asynchronous file filtering.
	stmt, err := c.tx.Prepare(`
		WITH RECURSIVE subtree(id, parent, sha1, level) AS (
			SELECT id, parent, sha1, 1 FROM node WHERE id = ?
			UNION ALL
			SELECT n.id, n.parent, n.sha1, s.level + 1
			FROM node AS n JOIN subtree AS s ON n.parent = s.id
		) DELETE FROM node WHERE id IN (
			SELECT id FROM subtree WHERE level <> 1 AND sha1 IS NULL
			AND id NOT IN (SELECT parent FROM node WHERE parent IS NOT NULL)
		)`)
	if err != nil {
		return err
	}

	for {
		if result, err := stmt.Exec(dbParent); err != nil {
			return err
		} else if n, err := result.RowsAffected(); err != nil {
			return err
		} else if n == 0 {
			return nil
		}
	}
}

type syncPath struct {
	db []string // database path, in terms of nodes
	fs string   // normalized filesystem path
}

// syncResolveRoots normalizes filesystem paths given in command line arguments,
// and figures out a database path for each. Duplicates are skipped or rejected.
func syncResolveRoots(args []string, fullpaths bool) (
	roots []*syncPath, err error) {
	for i := range args {
		fs, err := filepath.Abs(filepath.Clean(args[i]))
		if err != nil {
			return nil, err
		}

		roots = append(roots,
			&syncPath{decodeWebPath(filepath.ToSlash(fs)), fs})
	}

	if fullpaths {
		// Filter out duplicates. In this case, they're just duplicated work.
		slices.SortFunc(roots, func(a, b *syncPath) int {
			return strings.Compare(a.fs, b.fs)
		})
		roots = slices.CompactFunc(roots, func(a, b *syncPath) bool {
			if a.fs != b.fs && !strings.HasPrefix(b.fs, a.fs+"/") {
				return false
			}
			log.Printf("asking to sync path twice: %s\n", b.fs)
			return true
		})
	} else {
		// Keep just the basenames.
		for _, path := range roots {
			if len(path.db) > 0 {
				path.db = path.db[len(path.db)-1:]
			}
		}

		// Different filesystem paths mapping to the same DB location
		// are definitely a problem we would like to avoid,
		// otherwise we don't care.
		slices.SortFunc(roots, func(a, b *syncPath) int {
			return slices.Compare(a.db, b.db)
		})
		for i := 1; i < len(roots); i++ {
			if slices.Equal(roots[i-1].db, roots[i].db) {
				return nil, fmt.Errorf("duplicate root: %v", roots[i].db)
			}
		}
	}
	return
}

const disposeCTE = `WITH RECURSIVE
	root(id, sha1, parent, path) AS (
		SELECT id, sha1, parent, name FROM node WHERE id = ?
		UNION ALL
		SELECT r.id, r.sha1, n.parent, n.name || '/' || r.path
		FROM node AS n JOIN root AS r ON n.id = r.parent
	),
	children(id, sha1, path, level) AS (
		SELECT id, sha1, path, 1 FROM root WHERE parent IS NULL
		UNION ALL
		SELECT n.id, n.sha1, c.path || '/' || n.name, c.level + 1
		FROM node AS n JOIN children AS c ON n.parent = c.id
	),
	removed(sha1, count, path) AS (
		SELECT sha1, COUNT(*) AS count, MIN(path) AS path
		FROM children
		GROUP BY sha1
	),
	orphaned(sha1, path, count, total) AS (
		SELECT r.sha1, r.path, r.count, COUNT(*) AS total
		FROM removed AS r
		JOIN node ON node.sha1 = r.sha1
		GROUP BY node.sha1
		HAVING count = total
	)`

// cmdSync ensures the given (sub)roots are accurately reflected
// in the database.
func cmdSync(fs *flag.FlagSet, args []string) error {
	fullpaths := fs.Bool("fullpaths", false, "don't basename arguments")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 {
		return errWrongUsage
	}
	if err := openDB(fs.Arg(0)); err != nil {
		return err
	}

	roots, err := syncResolveRoots(fs.Args()[1:], *fullpaths)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// In case of a failure during processing, the only retained side effects
	// on the filesystem tree are:
	//  - Fixing dead symlinks to images.
	//  - Creating symlinks to images that aren't used by anything.
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Mild hack: upgrade the transaction to a write one straight away,
	// in order to rule out deadlocks (preventable failure).
	if _, err := tx.Exec(`END TRANSACTION;
		BEGIN IMMEDIATE TRANSACTION`); err != nil {
		return err
	}

	c := syncContext{ctx: ctx, tx: tx, pb: newProgressBar(-1),
		linked: make(map[string]struct{})}
	defer c.pb.Stop()

	if c.stmtOrphan, err = c.tx.Prepare(disposeCTE + `
		INSERT OR IGNORE INTO orphan(sha1, path)
		SELECT sha1, path FROM orphaned`); err != nil {
		return err
	}
	if c.stmtDisposeSub, err = c.tx.Prepare(disposeCTE + `
		DELETE FROM node WHERE id
		IN (SELECT DISTINCT id FROM children WHERE level <> 1)`); err != nil {
		return err
	}
	if c.stmtDisposeAll, err = c.tx.Prepare(disposeCTE + `
		DELETE FROM node WHERE id
		IN (SELECT DISTINCT id FROM children)`); err != nil {
		return err
	}

	// Info tasks take a position in the task semaphore channel.
	// then fill the info channel.
	//
	// Immediately after syncDequeue(), the info channel is empty,
	// but the semaphore might be full.
	//
	// By having at least one position in the info channel,
	// we allow at least one info task to run to semaphore release,
	// so that syncEnqueue() doesn't deadlock.
	//
	// By making it the same size as the semaphore,
	// the end of this function doesn't need to dequeue while waiting.
	// It also prevents goroutine leaks despite leaving them running--
	// once they finish their job, they're gone,
	// and eventually the info channel would get garbage collected.
	//
	// The additional slot is there to handle the one result
	// that may be placed while syncEnqueue() waits for the semaphore,
	// i.e., it is for the result of the task that syncEnqueue() spawns.
	c.info = make(chan syncFileInfo, cap(taskSemaphore)+1)

	for _, root := range roots {
		if err := syncRoot(&c, root.db, root.fs); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// --- Removal -----------------------------------------------------------------

// cmdRemove is for manual removal of subtrees from the database.
// Beware that inputs are database, not filesystem paths.
func cmdRemove(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 {
		return errWrongUsage
	}
	if err := openDB(fs.Arg(0)); err != nil {
		return err
	}

	tx, err := db.BeginTx(context.Background(), nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	for _, path := range fs.Args()[1:] {
		var id sql.NullInt64
		for _, name := range decodeWebPath(path) {
			if err := tx.QueryRow(`SELECT id FROM node
				WHERE parent IS ? AND name = ?`,
				id, name).Scan(&id); err != nil {
				return err
			}
		}
		if id.Int64 == 0 {
			return errors.New("can't remove root")
		}

		if _, err = tx.Exec(disposeCTE+`
			INSERT OR IGNORE INTO orphan(sha1, path)
			SELECT sha1, path FROM orphaned`, id); err != nil {
			return err
		}
		if _, err = tx.Exec(disposeCTE+`
			DELETE FROM node WHERE id
			IN (SELECT DISTINCT id FROM children)`, id); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// --- Tagging -----------------------------------------------------------------

// cmdTag mass imports tags from data passed on stdin as a TSV
// of SHA1 TAG WEIGHT entries.
func cmdTag(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.NArg() > 3 {
		return errWrongUsage
	}
	if err := openDB(fs.Arg(0)); err != nil {
		return err
	}

	space := fs.Arg(1)

	var description sql.NullString
	if fs.NArg() >= 3 {
		description = sql.NullString{String: fs.Arg(2), Valid: true}
	}

	// Note that starting as a write transaction prevents deadlocks.
	// Imports are rare, and just bulk load data, so this scope is fine.
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`INSERT OR IGNORE INTO tag_space(name, description)
		VALUES (?, ?)`, space, description); err != nil {
		return err
	}

	var spaceID int64
	if err := tx.QueryRow(`SELECT id FROM tag_space WHERE name = ?`,
		space).Scan(&spaceID); err != nil {
		return err
	}

	// XXX: It might make sense to pre-erase all tag assignments within
	// the given space for that image, the first time we see it:
	//
	//   DELETE FROM tag_assignment
	//   WHERE sha1 = ? AND tag IN (SELECT id FROM tag WHERE space = ?)
	//
	// or even just clear the tag space completely:
	//
	//   DELETE FROM tag_assignment
	//   WHERE tag IN (SELECT id FROM tag WHERE space = ?);
	//   DELETE FROM tag WHERE space = ?;
	stmt, err := tx.Prepare(`INSERT INTO tag_assignment(sha1, tag, weight)
		VALUES (?, (SELECT id FROM tag WHERE space = ? AND name = ?), ?)
		ON CONFLICT DO UPDATE SET weight = ?`)
	if err != nil {
		return err
	}

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		fields := strings.Split(scanner.Text(), "\t")
		if len(fields) != 3 {
			return errors.New("invalid input format")
		}

		sha1, tag := fields[0], fields[1]
		weight, err := strconv.ParseFloat(fields[2], 64)
		if err != nil {
			return err
		}

		if _, err := tx.Exec(
			`INSERT OR IGNORE INTO tag(space, name) VALUES (?, ?);`,
			spaceID, tag); err != nil {
			return nil
		}
		if _, err := stmt.Exec(sha1, spaceID, tag, weight, weight); err != nil {
			log.Printf("%s: %s\n", sha1, err)
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return tx.Commit()
}

// --- Check -------------------------------------------------------------------

func isValidSHA1(hash string) bool {
	if len(hash) != sha1.Size*2 || strings.ToLower(hash) != hash {
		return false
	}
	if _, err := hex.DecodeString(hash); err != nil {
		return false
	}
	return true
}

func hashesToFileListing(root, suffix string, hashes []string) []string {
	// Note that we're semi-duplicating {image,thumb}Path().
	paths := []string{root}
	for _, hash := range hashes {
		dir := filepath.Join(root, hash[:2])
		paths = append(paths, dir, filepath.Join(dir, hash+suffix))
	}
	slices.Sort(paths)
	return slices.Compact(paths)
}

func collectFileListing(root string) (paths []string, err error) {
	err = filepath.WalkDir(root,
		func(path string, d fs.DirEntry, err error) error {
			paths = append(paths, path)
			return err
		})

	// Even though it should already be sorted somehow.
	slices.Sort(paths)
	return
}

func checkFiles(root, suffix string, hashes []string) (bool, []string, error) {
	db := hashesToFileListing(root, suffix, hashes)
	fs, err := collectFileListing(root)
	if err != nil {
		return false, nil, err
	}

	iDB, iFS, ok, intersection := 0, 0, true, []string{}
	for iDB < len(db) && iFS < len(fs) {
		if db[iDB] == fs[iFS] {
			intersection = append(intersection, db[iDB])
			iDB++
			iFS++
		} else if db[iDB] < fs[iFS] {
			ok = false
			fmt.Printf("only in DB: %s\n", db[iDB])
			iDB++
		} else {
			ok = false
			fmt.Printf("only in FS: %s\n", fs[iFS])
			iFS++
		}
	}
	for _, path := range db[iDB:] {
		ok = false
		fmt.Printf("only in DB: %s\n", path)
	}
	for _, path := range fs[iFS:] {
		ok = false
		fmt.Printf("only in FS: %s\n", path)
	}
	return ok, intersection, nil
}

func checkHash(path string) (message string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return err.Error(), nil
	}
	defer f.Close()

	// We get 2 levels of parent directories in here, just filter them out.
	if fi, err := f.Stat(); err != nil {
		return err.Error(), nil
	} else if fi.IsDir() {
		return "", nil
	}

	hash := sha1.New()
	_, err = io.CopyBuffer(hash, f, make([]byte, 65536))
	if err != nil {
		return err.Error(), nil
	}

	sha1 := hex.EncodeToString(hash.Sum(nil))
	if sha1 != filepath.Base(path) {
		return fmt.Sprintf("mismatch, found %s", sha1), nil
	}
	return "", nil
}

func checkHashes(paths []string) (bool, error) {
	log.Println("checking image hashes")
	var failed atomic.Bool
	err := parallelize(paths, func(path string) (string, error) {
		message, err := checkHash(path)
		if message != "" {
			failed.Store(true)
		}
		return message, err
	})
	return !failed.Load(), err
}

// cmdCheck carries out various database consistency checks.
func cmdCheck(fs *flag.FlagSet, args []string) error {
	full := fs.Bool("full", false, "verify image hashes")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return errWrongUsage
	}
	if err := openDB(fs.Arg(0)); err != nil {
		return err
	}

	// Check if hashes are in the right format.
	log.Println("checking image hashes")

	allSHA1, err := dbCollectStrings(`SELECT sha1 FROM image`)
	if err != nil {
		return err
	}

	ok := true
	for _, hash := range allSHA1 {
		if !isValidSHA1(hash) {
			ok = false
			fmt.Printf("invalid image SHA1: %s\n", hash)
		}
	}

	// This is, rather obviously, just a strict subset.
	// Although it doesn't run in the same transaction.
	thumbSHA1, err := dbCollectStrings(`SELECT sha1 FROM image
		WHERE thumbw IS NOT NULL OR thumbh IS NOT NULL`)
	if err != nil {
		return err
	}

	// This somewhat duplicates {image,thumb}Path().
	log.Println("checking SQL against filesystem")
	okImages, intersection, err := checkFiles(
		filepath.Join(galleryDirectory, nameOfImageRoot), "", allSHA1)
	if err != nil {
		return err
	}

	okThumbs, _, err := checkFiles(
		filepath.Join(galleryDirectory, nameOfThumbRoot), ".webp", thumbSHA1)
	if err != nil {
		return err
	}
	if !okImages || !okThumbs {
		ok = false
	}

	log.Println("checking for dead symlinks")
	for _, path := range intersection {
		if _, err := os.Stat(path); err != nil {
			ok = false
			fmt.Printf("%s: %s\n", path, err)
		}
	}

	if *full {
		if ok2, err := checkHashes(intersection); err != nil {
			return err
		} else if !ok2 {
			ok = false
		}
	}

	if !ok {
		return errors.New("detected inconsistencies")
	}
	return nil
}

// --- Thumbnailing ------------------------------------------------------------

func identifyThumbnail(path string) (w, h int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	config, err := webp.DecodeConfig(f)
	if err != nil {
		return
	}
	return config.Width, config.Height, nil
}

func makeThumbnail(load bool, pathImage, pathThumb string) (
	w, h int, err error) {
	if load {
		if w, h, err = identifyThumbnail(pathThumb); err == nil {
			return
		}
	}

	thumbDirname, _ := filepath.Split(pathThumb)
	if err := os.MkdirAll(thumbDirname, 0755); err != nil {
		return 0, 0, err
	}

	// This is still too much, but it will be effective enough.
	memoryLimit := strconv.FormatInt(
		int64(C.sysconf(C._SC_PHYS_PAGES)*C.sysconf(C._SC_PAGE_SIZE))/
			int64(len(taskSemaphore)), 10)

	// Create a normalized thumbnail. Since we don't particularly need
	// any complex processing, such as surrounding metadata,
	// simply push it through ImageMagick.
	//
	//  - http://www.ericbrasseur.org/gamma.html
	//  - https://www.imagemagick.org/Usage/thumbnails/
	//  - https://imagemagick.org/script/command-line-options.php#layers
	//
	// "info:" output is written for each frame, which is why we delete
	// all of them but the first one beforehands.
	//
	// TODO: See if we can optimize resulting WebP animations.
	// (Do -layers optimize* apply to this format at all?)
	cmd := exec.Command("convert", "-limit", "thread", "1",

		// Do not invite the OOM killer, a particularly unpleasant guest.
		"-limit", "memory", memoryLimit,

		// ImageMagick creates files in /tmp, but that tends to be a tmpfs,
		// which is backed by memory. The path could also be moved elsewhere:
		// -define registry:temporary-path=/var/tmp
		"-limit", "map", "0", "-limit", "disk", "0",

		pathImage, "-coalesce", "-colorspace", "RGB", "-auto-orient", "-strip",
		"-resize", "256x128>", "-colorspace", "sRGB",
		"-format", "%w %h", "+write", pathThumb, "-delete", "1--1", "info:")

	out, err := cmd.Output()
	if err != nil {
		return 0, 0, err
	}

	_, err = fmt.Fscanf(bytes.NewReader(out), "%d %d", &w, &h)
	return w, h, err
}

// cmdThumbnail generates missing thumbnails, in parallel.
func cmdThumbnail(fs *flag.FlagSet, args []string) error {
	load := fs.Bool("load", false, "try to load existing thumbnail files")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return errWrongUsage
	}
	if err := openDB(fs.Arg(0)); err != nil {
		return err
	}

	hexSHA1 := fs.Args()[1:]
	if len(hexSHA1) == 0 {
		// Get all unique images in the database with no thumbnail.
		var err error
		hexSHA1, err = dbCollectStrings(`SELECT sha1 FROM image
			WHERE thumbw IS NULL OR thumbh IS NULL`)
		if err != nil {
			return err
		}
	}

	stmt, err := db.Prepare(
		`UPDATE image SET thumbw = ?, thumbh = ? WHERE sha1 = ?`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	var mu sync.Mutex
	return parallelize(hexSHA1, func(sha1 string) (message string, err error) {
		pathImage := imagePath(sha1)
		pathThumb := thumbPath(sha1)
		w, h, err := makeThumbnail(*load, pathImage, pathThumb)
		if err != nil {
			if ee, ok := err.(*exec.ExitError); ok {
				if message = string(ee.Stderr); message != "" {
					return message, nil
				}
				return ee.String(), nil
			}
			return "", err
		}

		mu.Lock()
		defer mu.Unlock()
		_, err = stmt.Exec(w, h, sha1)
		return "", err
	})
}

// --- Perceptual hash ---------------------------------------------------------

type linearImage struct {
	img image.Image
}

func newLinearImage(img image.Image) *linearImage {
	return &linearImage{img: img}
}

func (l *linearImage) ColorModel() color.Model { return l.img.ColorModel() }
func (l *linearImage) Bounds() image.Rectangle { return l.img.Bounds() }

func unSRGB(c uint32) uint8 {
	n := float64(c) / 0xffff
	if n <= 0.04045 {
		return uint8(n * (255.0 / 12.92))
	}
	return uint8(math.Pow((n+0.055)/(1.055), 2.4) * 255.0)
}

func (l *linearImage) At(x, y int) color.Color {
	r, g, b, a := l.img.At(x, y).RGBA()
	return color.RGBA{
		R: unSRGB(r), G: unSRGB(g), B: unSRGB(b), A: uint8(a >> 8)}
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// isWebPAnimation returns whether the given ReadSeeker starts a WebP animation.
// See https://developers.google.com/speed/webp/docs/riff_container
func isWebPAnimation(rs io.ReadSeeker) (bool, error) {
	b := make([]byte, 21)
	if _, err := rs.Read(b); err != nil {
		return false, err
	}
	if _, err := rs.Seek(0, io.SeekStart); err != nil {
		return false, err
	}

	return bytes.Equal(b[:4], []byte("RIFF")) &&
		bytes.Equal(b[8:16], []byte("WEBPVP8X")) &&
		b[20]&0b00000010 != 0, nil
}

var errIsAnimation = errors.New("cannot perceptually hash animations")

func dhashWebP(rs io.ReadSeeker) (uint64, error) {
	if a, err := isWebPAnimation(rs); err != nil {
		return 0, err
	} else if a {
		return 0, errIsAnimation
	}

	// Doing this entire thing in Go is SLOW, but convenient.
	source, err := webp.Decode(rs)
	if err != nil {
		return 0, err
	}

	var (
		linear  = newLinearImage(source)
		resized = image.NewNRGBA64(image.Rect(0, 0, 9, 8))
	)
	draw.CatmullRom.Scale(resized, resized.Bounds(),
		linear, linear.Bounds(), draw.Src, nil)

	var hash uint64
	for y := 0; y < 8; y++ {
		var grey [9]float32
		for x := 0; x < 9; x++ {
			rgba := resized.NRGBA64At(x, y)
			grey[x] = 0.2126*float32(rgba.R) +
				0.7152*float32(rgba.G) +
				0.0722*float32(rgba.B)
		}

		var row uint64
		if grey[0] < grey[1] {
			row |= 1 << 7
		}
		if grey[1] < grey[2] {
			row |= 1 << 6
		}
		if grey[2] < grey[3] {
			row |= 1 << 5
		}
		if grey[3] < grey[4] {
			row |= 1 << 4
		}
		if grey[4] < grey[5] {
			row |= 1 << 3
		}
		if grey[5] < grey[6] {
			row |= 1 << 2
		}
		if grey[6] < grey[7] {
			row |= 1 << 1
		}
		if grey[7] < grey[8] {
			row |= 1 << 0
		}
		hash = hash<<8 | row
	}
	return hash, nil
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

func makeDhash(sha1 string) (uint64, error) {
	pathThumb := thumbPath(sha1)
	f, err := os.Open(pathThumb)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return dhashWebP(f)
}

// cmdDhash computes perceptual hashes from thumbnails.
func cmdDhash(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return errWrongUsage
	}
	if err := openDB(fs.Arg(0)); err != nil {
		return err
	}

	hexSHA1 := fs.Args()[1:]
	if len(hexSHA1) == 0 {
		var err error
		hexSHA1, err = dbCollectStrings(`SELECT sha1 FROM image
			WHERE thumbw IS NOT NULL AND thumbh IS NOT NULL AND dhash IS NULL`)
		if err != nil {
			return err
		}
	}

	// Commits are very IO-expensive in both WAL and non-WAL SQLite,
	// so write this in one go. For a middle ground, we could batch the updates.
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Mild hack: upgrade the transaction to a write one straight away,
	// in order to rule out deadlocks (preventable failure).
	if _, err := tx.Exec(`END TRANSACTION;
		BEGIN IMMEDIATE TRANSACTION`); err != nil {
		return err
	}

	stmt, err := tx.Prepare(`UPDATE image SET dhash = ? WHERE sha1 = ?`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	var mu sync.Mutex
	err = parallelize(hexSHA1, func(sha1 string) (message string, err error) {
		hash, err := makeDhash(sha1)
		if errors.Is(err, errIsAnimation) {
			// Ignoring this common condition.
			return "", nil
		} else if err != nil {
			return err.Error(), nil
		}

		mu.Lock()
		defer mu.Unlock()
		_, err = stmt.Exec(int64(hash), sha1)
		return "", err
	})
	if err != nil {
		return err
	}
	return tx.Commit()
}

// --- Main --------------------------------------------------------------------

var errWrongUsage = errors.New("wrong usage")

var commands = map[string]struct {
	handler  func(*flag.FlagSet, []string) error
	usage    string
	function string
}{
	"init":      {cmdInit, "GD", "Initialize a database."},
	"web":       {cmdWeb, "GD ADDRESS", "Launch a web interface."},
	"tag":       {cmdTag, "GD SPACE [DESCRIPTION]", "Import tags."},
	"sync":      {cmdSync, "GD ROOT...", "Synchronise with the filesystem."},
	"remove":    {cmdRemove, "GD PATH...", "Remove database subtrees."},
	"check":     {cmdCheck, "GD", "Run consistency checks."},
	"thumbnail": {cmdThumbnail, "GD [SHA1...]", "Generate thumbnails."},
	"dhash":     {cmdDhash, "GD [SHA1...]", "Compute perceptual hashes."},
}

func usage() {
	f := flag.CommandLine.Output()
	fmt.Fprintf(f, "Usage: %s COMMAND [ARG...]\n", os.Args[0])
	flag.PrintDefaults()

	// The alphabetic ordering is unfortunate, but tolerable.
	keys := []string{}
	for key := range commands {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	fmt.Fprintf(f, "\nCommands:\n")
	for _, key := range keys {
		fmt.Fprintf(f, "  %s [OPTION...] %s\n    \t%s\n",
			key, commands[key].usage, commands[key].function)
	}
}

func main() {
	threads := flag.Int("threads", -1, "level of parallelization")

	// This implements the -h switch for us by default.
	// The rest of the handling here closely follows what flag does internally.
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(2)
	}

	cmd, ok := commands[flag.Arg(0)]
	if !ok {
		fmt.Fprintf(flag.CommandLine.Output(),
			"unknown command: %s\n", flag.Arg(0))
		flag.Usage()
		os.Exit(2)
	}

	fs := flag.NewFlagSet(flag.Arg(0), flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(),
			"Usage: %s [OPTION...] %s\n%s\n",
			fs.Name(), cmd.usage, cmd.function)
		fs.PrintDefaults()
	}

	if *threads > 0 {
		taskSemaphore = newSemaphore(*threads)
	} else {
		taskSemaphore = newSemaphore(runtime.NumCPU())
	}

	err := cmd.handler(fs, flag.Args()[1:])

	// Note that the database object has a closing finalizer,
	// we just additionally print any errors coming from there.
	if db != nil {
		if err := db.Close(); err != nil {
			log.Println(err)
		}
	}

	if errors.Is(err, errWrongUsage) {
		fs.Usage()
		os.Exit(2)
	} else if err != nil {
		log.Fatalln(err)
	}
}
