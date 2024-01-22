CREATE TABLE IF NOT EXISTS image(
	sha1     TEXT NOT NULL,     -- SHA-1 hash of file in lowercase hexadecimal
	width    INTEGER NOT NULL,  -- cached media width
	height   INTEGER NOT NULL,  -- cached media height
	thumbw   INTEGER,           -- cached thumbnail width, if known
	thumbh   INTEGER,           -- cached thumbnail height, if known
	dhash    INTEGER,           -- uint64 perceptual hash as a signed integer
	CHECK (unhex(sha1) IS NOT NULL AND lower(sha1) = sha1),
	PRIMARY KEY (sha1)
) STRICT;

CREATE INDEX IF NOT EXISTS image__dhash ON image(dhash);

--

CREATE TABLE IF NOT EXISTS node(
	id       INTEGER NOT NULL,             -- unique ID
	parent   INTEGER REFERENCES node(id),  -- root if NULL
	name     TEXT NOT NULL,                -- path component
	mtime    INTEGER,                      -- files: Unix time in seconds
	sha1     TEXT REFERENCES image(sha1),  -- files: content hash
	PRIMARY KEY (id)
) STRICT;

CREATE INDEX IF NOT EXISTS node__sha1 ON node(sha1);
CREATE UNIQUE INDEX IF NOT EXISTS node__parent_name
ON node(IFNULL(parent, 0), name);

CREATE TRIGGER IF NOT EXISTS node__sha1__check
BEFORE UPDATE OF sha1 ON node
WHEN OLD.sha1 IS NULL AND NEW.sha1 IS NOT NULL
AND EXISTS(SELECT id FROM node WHERE parent = OLD.id)
BEGIN
	SELECT RAISE(ABORT, 'trying to turn a non-empty directory into a file');
END;

/*
Automatic garbage collection, not sure if it actually makes any sense.
This needs PRAGMA recursive_triggers = 1; to work properly.

CREATE TRIGGER IF NOT EXISTS node__parent__gc
AFTER DELETE ON node FOR EACH ROW
BEGIN
	DELETE FROM node WHERE id = OLD.parent
	AND id NOT IN (SELECT DISTINCT parent FROM node);
END;
*/

--

CREATE TABLE IF NOT EXISTS orphan(
	sha1 TEXT NOT NULL REFERENCES image(sha1),
	path TEXT NOT NULL,  -- last occurence within the database hierarchy
	PRIMARY KEY (sha1)
) STRICT;

-- Renaming/moving a file can result either in a (ref, unref) or a (unref, ref)
-- sequence during sync, and I want to get at the same result.
CREATE TRIGGER IF NOT EXISTS node__sha1__deorphan_insert
AFTER INSERT ON node
WHEN NEW.sha1 IS NOT NULL
BEGIN
	DELETE FROM orphan WHERE sha1 = NEW.sha1;
END;

CREATE TRIGGER IF NOT EXISTS node__sha1__deorphan_update
AFTER UPDATE OF sha1 ON node
WHEN NEW.sha1 IS NOT NULL
BEGIN
	DELETE FROM orphan WHERE sha1 = NEW.sha1;
END;

--

CREATE TABLE IF NOT EXISTS tag_space(
	id          INTEGER NOT NULL,
	name        TEXT NOT NULL,
	description TEXT,
	CHECK (name NOT LIKE '%:%' AND name NOT LIKE '-%'),
	PRIMARY KEY (id)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS tag_space__name ON tag_space(name);

-- To avoid having to deal with NULLs, always create this special tag space.
INSERT OR IGNORE INTO tag_space(id, name, description)
VALUES(0, '', 'User-defined tags');

CREATE TABLE IF NOT EXISTS tag(
	id       INTEGER NOT NULL,
	space    INTEGER NOT NULL REFERENCES tag_space(id),
	name     TEXT NOT NULL,
	PRIMARY KEY (id)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS tag__space_name ON tag(space, name);

CREATE TABLE IF NOT EXISTS tag_assignment(
	sha1     TEXT NOT NULL REFERENCES image(sha1),
	tag      INTEGER NOT NULL REFERENCES tag(id),
	weight   REAL NOT NULL,     -- 0..1 normalized weight assigned to tag
	PRIMARY KEY (sha1, tag)
) STRICT;

CREATE INDEX IF NOT EXISTS tag_assignment__tag ON tag_assignment(tag);
