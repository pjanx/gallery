'use strict'

let callActive = false
let callFaulty = false

function call(method, params) {
	// XXX: At least with POST, unsuccessful requests result
	// in catched errors containing Errors with a null message.
	// This is an issue within XMLHttpRequest.
	callActive++
	return m.request({
		method: "POST",
		url: `/api/${method}`,
		body: params,
	}).then(result => {
		callActive--
		callFaulty = false
		return result
	}).catch(error => {
		callActive--
		callFaulty = true
		throw error
	})
}

const loading = (window.location.hostname !== 'localhost') ? 'lazy' : undefined

let Header = {
	global: [
		{name: "Browse",     route: '/browse'},
		{name: "Tags",       route: '/tags'},
		{name: "Duplicates", route: '/duplicates'},
		{name: "Orphans",    route: '/orphans'},
	],

	image: [
		{
			route: '/view',
			render: () => m(m.route.Link, {
				href: `/view/:key`,
				params: {key: m.route.param('key')},
				class: m.route.get().startsWith('/view')
					? 'active' : undefined,
			}, "View"),
		},
		{
			route: '/similar',
			render: () => m(m.route.Link, {
				href: `/similar/:key`,
				params: {key: m.route.param('key')},
				class: m.route.get().startsWith('/similar')
					? 'active' : undefined,
			}, "Similar"),
		},
	],

	search: [
		{
			route: '/search',
			render: () => m(m.route.Link, {
				href: `/search/:key`,
				params: {key: m.route.param('key')},
				class: m.route.get().startsWith('/search')
					? 'active' : undefined,
			}, "Search"),
		},
	],

	view(vnode) {
		const route = m.route.get()
		const main = this.global.map(x =>
			m(m.route.Link, {
				href: x.route,
				class: route.startsWith(x.route) ? 'active' : undefined,
			}, x.name))

		let context
		if (this.image.some(x => route.startsWith(x.route)))
			context = this.image.map(x => x.render())
		if (this.search.some(x => route.startsWith(x.route)))
			context = this.search.map(x => x.render())

		return m('.header', {}, [
			m('nav', main),
			m('nav', context),
			callFaulty
				? m('.activity.error[title=Error]', '●')
				: callActive
					? m('.activity[title=Busy]', '●')
					: m('.activity[title=Idle]', '○'),
		])
	},
}

let Thumbnail = {
	view(vnode) {
		const e = vnode.attrs.info
		if (!e.thumbW || !e.thumbH)
			return m('.thumbnail.missing', {...vnode.attrs, info: null})
		return m('img.thumbnail', {...vnode.attrs, info: null,
			src: `/thumb/${e.sha1}`, width: e.thumbW, height: e.thumbH,
			loading})
	},
}

let ScoredTag = {
	view(vnode) {
		const {space, tagname, score} = vnode.attrs
		return m('li', [
			m("meter[max=1.0]", {value: score, title: score}, score),
			` `,
			m(m.route.Link, {
				href: `/search/:key`,
				params: {key: `${space}:${tagname}`},
			}, ` ${tagname}`),
		])
	},
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

let BrowseModel = {
	path: undefined,
	subdirectories: [],
	entries: [],
	collator: new Intl.Collator(undefined, {numeric: true}),

	async reload(path) {
		if (this.path !== path) {
			this.path = path
			this.subdirectories = []
			this.entries = []
		}

		let resp = await call('browse', {path})
		this.subdirectories = resp.subdirectories
		this.entries = resp.entries.sort((a, b) =>
			this.collator.compare(a.name, b.name))
	},

	joinPath(parent, child) {
		if (!parent)
			return child
		if (!child)
			return parent
		return `${parent}/${child}`
	},

	getBrowseLinks() {
		if (this.path === undefined)
			return []

		let links = [{name: "Root", path: "", level: -1}], path
		for (const crumb of this.path.split('/').filter(s => !!s)) {
			path = this.joinPath(path, crumb)
			links.push({name: crumb, path: path, level: -1})
		}

		links[links.length - 1].level = 0

		for (const sub of this.subdirectories) {
			links.push(
				{name: sub, path: this.joinPath(this.path, sub), level: +1})
		}
		return links
	},
}

let BrowseBarLink = {
	view(vnode) {
		const link = vnode.attrs.link

		let c = 'selected'
		if (link.level < 0)
			c = 'parent'
		if (link.level > 0)
			c = 'child'

		return m('li', {
			class: c,
		}, m(m.route.Link, {
			href: `/browse/:key`,
			params: {key: link.path},
		}, link.name))
	},
}

let BrowseView = {
	// So that Page Up/Down, etc., work after changing directories.
	// Programmatically focusing a scrollable element requires setting tabindex,
	// and causes :focus-visible on page load, which we suppress in CSS.
	// I wish there was another way, but the workaround isn't particularly bad.
	// focus({focusVisible: true}) is FF 104+ only and experimental.
	oncreate(vnode) { vnode.dom.focus() },

	view(vnode) {
		return m('.browser[tabindex=0]', {
			// Trying to force the oncreate on path changes.
			key: BrowseModel.path,
		}, BrowseModel.entries.map(info => {
			return m(m.route.Link, {href: `/view/${info.sha1}`},
				m(Thumbnail, {info, title: info.name}))
		}))
	},
}

let Browse = {
	// Reload the model immediately, to improve responsivity.
	// But we don't need to: https://mithril.js.org/route.html#preloading-data
	// Also see: https://mithril.js.org/route.html#route-cancellation--blocking
	oninit(vnode) {
		let path = vnode.attrs.key || ""
		BrowseModel.reload(path)
	},

	view(vnode) {
		return m('.container', {}, [
			m(Header),
			m('.body', {}, [
				m('.sidebar', [
					m('ul.path', BrowseModel.getBrowseLinks()
						.map(link => m(BrowseBarLink, {link}))),
				]),
				m(BrowseView),
			]),
		])
	},
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

let TagsModel = {
	ns: null,
	namespaces: {},

	async reload(ns) {
		if (this.ns !== ns) {
			this.ns = ns
			this.namespaces = {}
		}

		this.namespaces = await call('tags', {namespace: ns})
	},
}

let TagsList = {
	view(vnode) {
		// TODO: Make it possible to sort by count.
		const tags = Object.entries(vnode.attrs.tags)
			.sort(([a, b]) => a[0].localeCompare(b[0]))

		return (tags.length == 0)
			? "No tags"
			: m("ul", tags.map(([name, count]) => m("li", [
				m(m.route.Link, {
					href: `/search/:key`,
					params: {key: `${vnode.attrs.space}:${name}`},
				}, ` ${name}`),
				` ×${count}`,
			])))
	},
}

let TagsView = {
	// See BrowseView.
	oncreate(vnode) { vnode.dom.focus() },

	view(vnode) {
		// XXX: The empty-named tag namespace gets a bit shafted,
		// in particular in the router, as well as with its header.
		// Maybe we could refer to it by its numeric ID in routing.
		const names = Object.keys(TagsModel.namespaces)
			.sort((a, b) => a.localeCompare(b))

		let children = (names.length == 0)
			? "No namespaces"
			: names.map(space => {
				const ns = TagsModel.namespaces[space]
				return [
					m("h2", space),
					ns.description ? m("p", ns.description) : [],
					m(TagsList, {space, tags: ns.tags}),
				]
			})
		return m('.tags[tabindex=0]', {}, children)
	},
}

let Tags = {
	oninit(vnode) {
		let ns = vnode.attrs.key
		TagsModel.reload(ns)
	},

	view(vnode) {
		return m('.container', {}, [
			m(Header),
			m('.body', {}, m(TagsView)),
		])
	},
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

let DuplicatesModel = {
	entries: [],

	async reload() {
		this.entries = await call('duplicates', {})
	},
}

let DuplicatesThumbnail = {
	view(vnode) {
		const info = vnode.attrs.info
		return [
			m(m.route.Link, {href: `/similar/${info.sha1}`},
				m(Thumbnail, {info})),
			(info.occurences != 1) ? ` ×${info.occurences}` : [],
		]
	},
}

let DuplicatesList = {
	// See BrowseView.
	oncreate(vnode) { vnode.dom.focus() },

	view(vnode) {
		let children = (DuplicatesModel.entries.length == 0)
			? "No duplicates"
			: DuplicatesModel.entries.map(group =>
				m('.row', group.map(entry =>
					m(DuplicatesThumbnail, {info: entry}))))
		return m('.duplicates[tabindex=0]', {}, children)
	},
}

let Duplicates = {
	oninit(vnode) {
		DuplicatesModel.reload()
	},

	view(vnode) {
		return m('.container', {}, [
			m(Header),
			m('.body', {}, m(DuplicatesList)),
		])
	},
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

let OrphansModel = {
	entries: [],

	async reload() {
		this.entries = await call('orphans', {})
	},
}

let OrphansReplacement = {
	view(vnode) {
		const info = vnode.attrs.info
		if (!info)
			return []

		return [
			` → `,
			m(m.route.Link, {href: `/view/${info.sha1}`},
				m(Thumbnail, {info})),
			`${info.tags} tags`,
		]
	},
}

let OrphansRow = {
	view(vnode) {
		const info = vnode.attrs.info
		return m('.row', [
			// It might not load, but still allow tag viewing.
			m(m.route.Link, {href: `/view/${info.sha1}`},
				m(Thumbnail, {info})),
			`${info.tags} tags`,
			m(OrphansReplacement, {info: info.replacement}),
		])
	},
}

let OrphansList = {
	// See BrowseView.
	oncreate(vnode) { vnode.dom.focus() },

	view(vnode) {
		let children = (OrphansModel.entries.length == 0)
			? "No orphans"
			: OrphansModel.entries.map(info => [
				m("h2", info.lastPath),
				m(OrphansRow, {info}),
			])
		return m('.orphans[tabindex=0]', {}, children)
	},
}

let Orphans = {
	oninit(vnode) {
		OrphansModel.reload()
	},

	view(vnode) {
		return m('.container', {}, [
			m(Header),
			m('.body', {}, m(OrphansList)),
		])
	},
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

let ViewModel = {
	sha1: undefined,
	width: 0,
	height: 0,
	paths: [],
	tags: {},

	async reload(sha1) {
		if (this.sha1 !== sha1) {
			this.sha1 = sha1
			this.width = this.height = 0
			this.paths = []
			this.tags = {}
		}

		let resp = await call('info', {sha1: sha1})
		this.width = resp.width
		this.height = resp.height
		this.paths = resp.paths
		this.tags = resp.tags
	},
}

let ViewBarBrowseLink = {
	view(vnode) {
		return m(m.route.Link, {
			href: `/browse/:key`,
			params: {key: vnode.attrs.path},
		}, vnode.attrs.name)
	},
}

let ViewBarPath = {
	view(vnode) {
		const parents = vnode.attrs.path.split('/')
		const basename = parents.pop()

		let result = [], path
		if (!parents.length)
			result.push(m(ViewBarBrowseLink, {path: "", name: "Root"}), "/")
		for (const crumb of parents) {
			path = BrowseModel.joinPath(path, crumb)
			result.push(m(ViewBarBrowseLink, {path, name: crumb}), "/")
		}
		result.push(basename)
		return result
	},
}

let ViewBar = {
	view(vnode) {
		return m('.viewbar', [
			m('h2', "Locations"),
			m('ul', ViewModel.paths.map(path =>
				m('li', m(ViewBarPath, {path})))),
			m('h2', "Tags"),
			Object.entries(ViewModel.tags).map(([space, tags]) =>
				m('details[open]', [
					m('summary', m("h3",
						m(m.route.Link, {href: `/tags/${space}`}, space))),
					m("ul.tags", Object.entries(tags)
						.sort(([t1, w1], [t2, w2]) => (w2 - w1))
						.map(([tag, score]) =>
							m(ScoredTag, {space, tagname: tag, score}))),
				])),
		])
	},
}

let View = {
	oninit(vnode) {
		let sha1 = vnode.attrs.key || ""
		ViewModel.reload(sha1)
	},

	view(vnode) {
		const view = m('.view', [
			ViewModel.sha1 !== undefined
				? m('img', {src: `/image/${ViewModel.sha1}`,
					width: ViewModel.width, height: ViewModel.height})
				: "No image.",
		])
		return m('.container', {}, [
			m(Header),
			m('.body', {}, [view, m(ViewBar)]),
		])
	},
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

let SimilarModel = {
	sha1: undefined,
	info: {paths: []},
	groups: {},

	async reload(sha1) {
		if (this.sha1 !== sha1) {
			this.sha1 = sha1
			this.info = {paths: []}
			this.groups = {}
		}

		let resp = await call('similar', {sha1: sha1})
		this.info = resp.info
		this.groups = resp.groups
	},
}

let SimilarThumbnail = {
	view(vnode) {
		const info = vnode.attrs.info
		return m(m.route.Link, {href: `/view/${info.sha1}`},
			m(Thumbnail, {info}))
	},
}

let SimilarGroup = {
	view(vnode) {
		const images = vnode.attrs.images
		let result = [
			m('h2', vnode.attrs.name),
			images.map(info => m('.row', [
				m(SimilarThumbnail, {info}),
				m('ul', [
					m('li', Math.round(info.pixelsRatio * 100) +
						"% pixels of input image"),
					info.paths.map(path =>
						m('li', m(ViewBarPath, {path}))),
				]),
			]))
		]
		if (!images.length)
			result.push("No matches.")
		return result
	},
}

let SimilarList = {
	view(vnode) {
		if (SimilarModel.sha1 === undefined ||
			SimilarModel.info.paths.length == 0)
			return "No image"

		const info = SimilarModel.info
		return m('.similar', {}, [
			m('.row', [
				m(SimilarThumbnail, {info}),
				m('ul', info.paths.map(path =>
					m('li', m(ViewBarPath, {path})))),
			]),
			Object.entries(SimilarModel.groups).map(([name, images]) =>
				m(SimilarGroup, {name, images})),
		])
	},
}

let Similar = {
	oninit(vnode) {
		let sha1 = vnode.attrs.key || ""
		SimilarModel.reload(sha1)
	},

	view(vnode) {
		return m('.container', {}, [
			m(Header),
			m('.body', {}, m(SimilarList)),
		])
	},
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

let SearchModel = {
	query: undefined,
	matches: [],
	related: {},

	async reload(query) {
		if (this.query !== query) {
			this.query = query
			this.matches = []
			this.related = {}
		}

		let resp = await call('search', {query})
		this.matches = resp.matches
		this.related = resp.related
	},
}

let SearchRelated = {
	view(vnode) {
		return Object.entries(SearchModel.related)
			.sort((a, b) => a[0].localeCompare(b[0]))
			.map(([space, tags]) => m('details[open]', [
				m('summary', m('h2',
					m(m.route.Link, {href: `/tags/${space}`}, space))),
				m('ul.tags', tags
					.sort((a, b) => (b.score - a.score))
					.map(({tag, score}) =>
						m(ScoredTag, {space, tagname: tag, score}))),
			]))
	},
}

let SearchView = {
	// See BrowseView.
	oncreate(vnode) { vnode.dom.focus() },

	view(vnode) {
		return m('.browser[tabindex=0]', {
			// Trying to force the oncreate on path changes.
			key: SearchModel.path,
		}, SearchModel.matches
			.sort((a, b) => b.score - a.score)
			.map(info => {
				return m(m.route.Link, {href: `/view/${info.sha1}`},
					m(Thumbnail, {info, title: info.score}))
			}))
	},
}

let Search = {
	oninit(vnode) {
		SearchModel.reload(vnode.attrs.key)
	},

	view(vnode) {
		return m('.container', {}, [
			m(Header),
			m('.body', {}, [
				m('.sidebar', [
					m('input', {
						value: SearchModel.query,
						onchange: event => m.route.set(
							`/search/:key`, {key: event.target.value}),
					}),
					m(SearchRelated),
				]),
				m(SearchView),
			]),
		])
	},
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

window.addEventListener('load', () => {
	m.route(document.body, "/browse/", {
		// The path doesn't need to be escaped, perhaps change that (":key...").
		"/browse/": Browse,
		"/browse/:key": Browse,
		"/tags": Tags,
		"/tags/:key": Tags,
		"/duplicates": Duplicates,
		"/orphans": Orphans,

		"/view/:key": View,
		"/similar/:key": Similar,

		"/search/:key": Search,
	})
})
