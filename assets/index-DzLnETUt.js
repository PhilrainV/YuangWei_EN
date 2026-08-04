import { r as u, j as e, c as C } from "./react-vendor-BTWVIjLd.js";
import { w as N } from "./world-map-CPvcksDd.js";
import {
  F as S,
  a as E,
  b as k,
  c as A,
  d as P,
  e as y,
  S as M,
} from "./icons-vendor-CUxSbwz5.js";
(function () {
  const d = document.createElement("link").relList;
  if (d && d.supports && d.supports("modulepreload")) return;
  for (const t of document.querySelectorAll('link[rel="modulepreload"]')) p(t);
  new MutationObserver((t) => {
    for (const h of t)
      if (h.type === "childList")
        for (const i of h.addedNodes)
          i.tagName === "LINK" && i.rel === "modulepreload" && p(i);
  }).observe(document, { childList: !0, subtree: !0 });
  function l(t) {
    const h = {};
    return (
      t.integrity && (h.integrity = t.integrity),
      t.referrerPolicy && (h.referrerPolicy = t.referrerPolicy),
      t.crossOrigin === "use-credentials"
        ? (h.credentials = "include")
        : t.crossOrigin === "anonymous"
          ? (h.credentials = "omit")
          : (h.credentials = "same-origin"),
      h
    );
  }
  function p(t) {
    if (t.ep) return;
    t.ep = !0;
    const h = l(t);
    fetch(t.href, h);
  }
})();
const m = window.YUANG_WEI_CONTENT ?? {};
function f(n) {
  return Array.isArray(n) ? n : [];
}
const U = f(m.publications),
  o = m.profile,
  $ = f(m.books),
  F = f(m.patents),
  L = f(m.softwareCopyrights),
  O = f(m.honors),
  V = f(m.experience),
  w = "https://yuang-wei-academic.philrain-cs.chatgpt.site/api/visitors",
  B =
    "https://raw.githubusercontent.com/PhilrainV/YuangWei_EN/google-scholar-stats/gs_data.json",
  I = [
    ["About", "about"],
    ["Education", "education"],
    ["Publications", "publications"],
    ["Other Outputs", "outputs"],
    ["Honors", "honors"],
    ["Experience", "experience"],
  ],
  W = new Intl.DisplayNames(["en"], { type: "region" });
function Y(n) {
  const d =
    n.countryCode && n.countryCode !== "XX"
      ? W.of(n.countryCode)
      : "Unknown location";
  return [n.city, d].filter(Boolean).join(", ");
}
function b({ text: n }) {
  const d = n.split(/(Yuang Wei|Yu’ang Wei|Wei Yuang|魏雨昂)/g);
  return e.jsx(e.Fragment, {
    children: d.map((l, p) =>
      /^(Yuang Wei|Yu’ang Wei|Wei Yuang|魏雨昂)$/.test(l)
        ? e.jsx(
            "strong",
            { className: "author-self", children: l },
            `${l}-${p}`,
          )
        : e.jsx("span", { children: l }, `${l}-${p}`),
    ),
  });
}
function j({ id: n, children: d }) {
  return e.jsx("h2", { className: "section-heading", id: n, children: d });
}
function c({ href: n, children: d, className: l = "" }) {
  return e.jsx("a", {
    className: l,
    href: n,
    target: "_blank",
    rel: "noreferrer",
    children: d,
  });
}
function T() {
  const [n, d] = u.useState(null),
    [l, p] = u.useState(!1);
  u.useEffect(() => {
    let i = !0;
    async function x() {
      try {
        await fetch(w, { method: "POST", mode: "cors" });
        const g = await fetch(w, { cache: "no-store", mode: "cors" });
        if (!g.ok) throw new Error("visitor statistics unavailable");
        const v = await g.json();
        i && d(v);
      } catch {
        i && p(!0);
      }
    }
    return (
      x(),
      () => {
        i = !1;
      }
    );
  }, []);
  const t = u.useMemo(
      () => new Map((n?.countries ?? []).map((i) => [i.code.toLowerCase(), i])),
      [n],
    ),
    h = Math.max(1, ...(n?.countries ?? []).map((i) => i.visits));
  return e.jsxs("div", {
    className: "visitor-dashboard",
    children: [
      e.jsxs("div", {
        className: "visitor-map-panel",
        children: [
          e.jsx("svg", {
            className: "world-map",
            viewBox: N.viewBox,
            role: "img",
            "aria-label": "World map of website visitors",
            children: N.locations.map((i) => {
              const x = t.get(i.id),
                g = x ? 0.3 + (x.visits / h) * 0.7 : 0;
              return e.jsx(
                "path",
                {
                  className: x ? "country-shape has-visits" : "country-shape",
                  d: i.path,
                  style: x ? { opacity: g } : void 0,
                  children: e.jsx("title", {
                    children: x ? `${i.name}: ${x.visits} visits` : i.name,
                  }),
                },
                i.id,
              );
            }),
          }),
          e.jsx("p", {
            className: "map-credit",
            children: "Map data: SVG Maps (CC BY 4.0)",
          }),
        ],
      }),
      e.jsxs("div", {
        className: "visitor-summary",
        "aria-live": "polite",
        children: [
          e.jsxs("div", {
            className: "visitor-metrics",
            children: [
              e.jsxs("div", {
                children: [
                  e.jsx("strong", { children: n?.totalVisits ?? "—" }),
                  e.jsx("span", { children: "Total visits" }),
                ],
              }),
              e.jsxs("div", {
                children: [
                  e.jsx("strong", { children: n?.uniqueVisitors ?? "—" }),
                  e.jsx("span", { children: "Unique visitors" }),
                ],
              }),
              e.jsxs("div", {
                children: [
                  e.jsx("strong", { children: n?.countries.length ?? "—" }),
                  e.jsx("span", { children: "Countries / regions" }),
                ],
              }),
            ],
          }),
          e.jsx("h3", { children: "Recent visitors" }),
          l
            ? e.jsx("p", {
                className: "visitor-empty",
                children: "Visitor statistics are temporarily unavailable.",
              })
            : n?.recentVisitors.length
              ? e.jsx("ol", {
                  className: "recent-visitors",
                  children: n.recentVisitors.map((i) =>
                    e.jsxs(
                      "li",
                      {
                        children: [
                          e.jsxs("span", {
                            className: "visitor-id",
                            children: ["Visitor ", i.id],
                          }),
                          e.jsx("span", { children: Y(i) }),
                        ],
                      },
                      `${i.id}-${i.lastSeen}`,
                    ),
                  ),
                })
              : e.jsx("p", {
                  className: "visitor-empty",
                  children: "Loading visitor data…",
                }),
          e.jsx("p", {
            className: "privacy-note",
            children:
              "To protect visitor privacy, only an anonymous identifier and approximate location are displayed. Full IP addresses are never shown.",
          }),
        ],
      }),
    ],
  });
}
function _() {
  const [n, d] = u.useState(!1),
    [l, p] = u.useState("All"),
    [t, h] = u.useState(!1),
    [i, x] = u.useState({
      citedby: 330,
      hindex: 11,
      i10index: 12,
      updated: "2026-08-02",
    });
  u.useEffect(() => {
    let s = !0;
    return (
      fetch(B, { cache: "no-store" })
        .then((r) => {
          if (!r.ok) throw new Error("scholar statistics unavailable");
          return r.json();
        })
        .then((r) => {
          s &&
            x((a) => ({
              citedby: Number(r.citedby ?? a.citedby),
              hindex: Number(r.hindex ?? a.hindex),
              i10index: Number(r.i10index ?? a.i10index),
              updated: r.updated ?? a.updated,
            }));
        })
        .catch(() => {}),
      () => {
        s = !1;
      }
    );
  }, []);
  const g = u.useMemo(
    () =>
      [
        "Journal Articles",
        "Chinese-Language Journal Articles",
        "Conference Papers",
      ]
        .filter(
          (r) =>
            l === "All" ||
            (l === "Journals"
              ? r !== "Conference Papers"
              : r === "Conference Papers"),
        )
        .map((r) => ({ group: r, papers: U.filter((a) => a.group === r) })),
    [l],
  );
  async function v() {
    try {
      await navigator.clipboard.writeText(o.email);
    } catch {
      const s = document.createElement("textarea");
      ((s.value = o.email),
        (s.style.position = "fixed"),
        (s.style.opacity = "0"),
        document.body.appendChild(s),
        s.select(),
        document.execCommand("copy"),
        s.remove());
    }
    (h(!0), window.setTimeout(() => h(!1), 1600));
  }
  return e.jsxs(e.Fragment, {
    children: [
      e.jsx("header", {
        className: "site-header",
        children: e.jsxs("div", {
          className: "header-inner",
          children: [
            e.jsx("a", {
              className: "site-title",
              href: "#about",
              "aria-label": "Back to the top",
              children: o.name,
            }),
            e.jsxs("button", {
              className: "menu-button",
              type: "button",
              "aria-label": n ? "Close navigation" : "Open navigation",
              "aria-expanded": n,
              onClick: () => d((s) => !s),
              children: [
                e.jsx("span", {}),
                e.jsx("span", {}),
                e.jsx("span", {}),
              ],
            }),
            e.jsxs("nav", {
              className: n ? "main-nav is-open" : "main-nav",
              "aria-label": "Main navigation",
              children: [
                I.map(([s, r]) =>
                  e.jsx(
                    "a",
                    { href: `#${r}`, onClick: () => d(!1), children: s },
                    r,
                  ),
                ),
                e.jsxs("a", {
                  className: "language-switch",
                  href: "https://philrainv.github.io/",
                  target: "_self",
                  "aria-label": "Switch to the Chinese homepage",
                  children: [e.jsx(S, { "aria-hidden": "true" }), "中文"],
                }),
              ],
            }),
          ],
        }),
      }),
      e.jsxs("main", {
        className: "page-shell",
        children: [
          e.jsxs("aside", {
            className: "profile-panel",
            "aria-label": "Profile",
            children: [
              e.jsx("img", {
                className: "portrait",
                src: o.avatar,
                alt: o.avatarAlt,
              }),
              e.jsxs("div", {
                className: "profile-intro",
                children: [
                  e.jsx("h1", { children: o.name }),
                  e.jsxs("p", {
                    className: "position",
                    children: [o.affiliation, " · ", o.title],
                  }),
                ],
              }),
              e.jsxs("div", {
                className: "contact-list",
                children: [
                  e.jsxs("div", {
                    className: "contact-location",
                    children: [
                      e.jsx(E, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      o.location,
                    ],
                  }),
                  e.jsxs("button", {
                    type: "button",
                    onClick: v,
                    children: [
                      e.jsx(k, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      t ? "Email copied" : o.email,
                    ],
                  }),
                  e.jsxs(c, {
                    href: o.links.researchGate,
                    children: [
                      e.jsx(A, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "ResearchGate",
                    ],
                  }),
                  e.jsxs(c, {
                    href: o.links.github,
                    children: [
                      e.jsx(P, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "GitHub",
                    ],
                  }),
                  e.jsxs(c, {
                    href: o.links.googleScholar,
                    children: [
                      e.jsx(y, {
                        className: "contact-icon scholar-mark",
                        "aria-hidden": "true",
                      }),
                      "Google Scholar",
                    ],
                  }),
                  e.jsxs(c, {
                    href: o.links.orcid,
                    children: [
                      e.jsx(M, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "ORCID",
                    ],
                  }),
                ],
              }),
              e.jsxs(c, {
                href: o.links.googleScholar,
                className: "scholar-card",
                children: [
                  e.jsxs("div", {
                    className: "scholar-card-title",
                    children: [
                      e.jsxs("span", {
                        children: [
                          e.jsx(y, { "aria-hidden": "true" }),
                          " Google Scholar",
                        ],
                      }),
                      e.jsx("span", { "aria-hidden": "true", children: "↗" }),
                    ],
                  }),
                  e.jsxs("div", {
                    className: "scholar-metrics",
                    children: [
                      e.jsxs("div", {
                        children: [
                          e.jsx("strong", { children: i.citedby }),
                          e.jsx("span", { children: "Citations" }),
                        ],
                      }),
                      e.jsxs("div", {
                        children: [
                          e.jsx("strong", { children: i.hindex }),
                          e.jsx("span", { children: "h-index" }),
                        ],
                      }),
                      e.jsxs("div", {
                        children: [
                          e.jsx("strong", { children: i.i10index }),
                          e.jsx("span", { children: "i10-index" }),
                        ],
                      }),
                    ],
                  }),
                  e.jsxs("small", {
                    children: [
                      "Automatically updated",
                      i.updated
                        ? ` · ${i.updated.slice(0, 10).replaceAll("-", ".")}`
                        : "",
                    ],
                  }),
                ],
              }),
            ],
          }),
          e.jsxs("div", {
            className: "main-content",
            children: [
              e.jsxs("section", {
                className: "content-section about-section",
                "aria-labelledby": "about",
                children: [
                  e.jsx(j, { id: "about", children: "About Me" }),
                  e.jsx("div", {
                    className: "intro-text",
                    children: o.bio.map((s) => e.jsx("p", { children: s }, s)),
                  }),
                  e.jsxs("div", {
                    className: "research-row",
                    children: [
                      e.jsx("strong", { children: "Research" }),
                      e.jsx("div", {
                        children: o.researchInterests.map((s) =>
                          e.jsx("span", { children: s }, s),
                        ),
                      }),
                    ],
                  }),
                ],
              }),
              e.jsxs("section", {
                className: "content-section education-section",
                "aria-labelledby": "education",
                children: [
                  e.jsx(j, { id: "education", children: "Education" }),
                  e.jsxs("div", {
                    className: "education-list",
                    children: [
                      e.jsxs("article", {
                        className: "education-item",
                        children: [
                          e.jsx(c, {
                            href: "https://aiedu.ecnu.edu.cn/",
                            className: "school-logo-link",
                            children: e.jsx("img", {
                              src: "images/ECNU_logo.png",
                              alt: "East China Normal University logo",
                            }),
                          }),
                          e.jsx("div", {
                            className: "education-time",
                            children: "2022.06 — 2026.06",
                          }),
                          e.jsxs("div", {
                            className: "education-body",
                            children: [
                              e.jsx("h3", {
                                children: e.jsx(c, {
                                  href: "https://aiedu.ecnu.edu.cn/",
                                  children:
                                    "East China Normal University (ECNU)",
                                }),
                              }),
                              e.jsx("p", {
                                children:
                                  "Shanghai Institute of AI for Education · Intelligent Education · Ph.D.",
                              }),
                              e.jsxs("p", {
                                className: "education-note",
                                children: [
                                  "Supervisor:",
                                  " ",
                                  e.jsx(c, {
                                    href: "https://faculty.ecnu.edu.cn/_s8/jb2/main.psp",
                                    children: "Prof. Bo Jiang",
                                  }),
                                  e.jsx("span", {
                                    children: "Shanghai, China",
                                  }),
                                ],
                              }),
                            ],
                          }),
                        ],
                      }),
                      e.jsxs("article", {
                        className: "education-item",
                        children: [
                          e.jsx(c, {
                            href: "https://www.comp.nus.edu.sg/cs/",
                            className: "school-logo-link",
                            children: e.jsx("img", {
                              src: "images/NUS_logo.png",
                              alt: "National University of Singapore logo",
                            }),
                          }),
                          e.jsx("div", {
                            className: "education-time",
                            children: "2024.09 — 2025.09",
                          }),
                          e.jsxs("div", {
                            className: "education-body",
                            children: [
                              e.jsx("h3", {
                                children: e.jsx(c, {
                                  href: "https://www.comp.nus.edu.sg/cs/",
                                  children:
                                    "National University of Singapore (NUS)",
                                }),
                              }),
                              e.jsx("p", {
                                children:
                                  "Department of Computer Science · Human–Computer Interaction · CSC Visiting Ph.D. Student",
                              }),
                              e.jsxs("p", {
                                className: "education-note",
                                children: [
                                  "Supervisor:",
                                  " ",
                                  e.jsx(c, {
                                    href: "https://www.comp.nus.edu.sg/cs/people/brianlim/",
                                    children: "Assoc. Prof. Brian Y. Lim",
                                  }),
                                  e.jsx("span", { children: "Singapore" }),
                                ],
                              }),
                            ],
                          }),
                        ],
                      }),
                      e.jsxs("article", {
                        className: "education-item",
                        children: [
                          e.jsx(c, {
                            href: "https://dqgc.ncut.edu.cn/",
                            className: "school-logo-link",
                            children: e.jsx("img", {
                              src: "images/NCUT_logo.png",
                              alt: "North China University of Technology logo",
                            }),
                          }),
                          e.jsx("div", {
                            className: "education-time",
                            children: "2015.09 — 2022.06",
                          }),
                          e.jsxs("div", {
                            className: "education-body",
                            children: [
                              e.jsx("h3", {
                                children: e.jsx(c, {
                                  href: "https://dqgc.ncut.edu.cn/",
                                  children:
                                    "North China University of Technology (NCUT)",
                                }),
                              }),
                              e.jsx("p", {
                                children:
                                  "School of Electrical and Control Engineering · B.Eng. in Automation and M.Eng. in Control Science and Engineering",
                              }),
                              e.jsxs("p", {
                                className: "education-note",
                                children: [
                                  "Supervisor:",
                                  " ",
                                  e.jsx(c, {
                                    href: "https://dqgc.ncut.edu.cn/info/1228/3137.htm",
                                    children: "Assoc. Prof. Jining Xu",
                                  }),
                                  e.jsx("span", { children: "Beijing, China" }),
                                ],
                              }),
                            ],
                          }),
                        ],
                      }),
                    ],
                  }),
                ],
              }),
              e.jsxs("section", {
                className: "content-section",
                "aria-labelledby": "publications",
                children: [
                  e.jsxs("div", {
                    className: "heading-with-tools",
                    children: [
                      e.jsx(j, {
                        id: "publications",
                        children: "Publications",
                      }),
                      e.jsx("div", {
                        className: "paper-filter",
                        "aria-label": "Filter publications by type",
                        children: ["All", "Journals", "Conferences"].map((s) =>
                          e.jsx(
                            "button",
                            {
                              type: "button",
                              className: l === s ? "is-active" : "",
                              "aria-pressed": l === s,
                              onClick: () => p(s),
                              children: s,
                            },
                            s,
                          ),
                        ),
                      }),
                    ],
                  }),
                  e.jsxs("p", {
                    className: "publication-legend",
                    children: [
                      e.jsx("span", { children: "Yuang Wei" }),
                      " indicates my name; publications are grouped by type and ordered by year.",
                    ],
                  }),
                  g.map(({ group: s, papers: r }) =>
                    e.jsxs(
                      "div",
                      {
                        className: "publication-group",
                        children: [
                          e.jsxs("h3", {
                            className: "publication-group-title",
                            children: [
                              s,
                              e.jsx("span", { children: r.length }),
                            ],
                          }),
                          e.jsx("div", {
                            className: "publication-list",
                            children: r.map((a) =>
                              e.jsxs(
                                "article",
                                {
                                  className: a.image
                                    ? "publication-item with-image"
                                    : "publication-item",
                                  children: [
                                    a.image &&
                                      e.jsx("img", {
                                        className: "publication-image",
                                        src: a.image,
                                        alt: `Illustration for ${a.title}`,
                                        loading: "lazy",
                                        style: {
                                          objectFit: a.imageFit || "cover",
                                        },
                                      }),
                                    e.jsxs("div", {
                                      className: "publication-main",
                                      children: [
                                        e.jsxs("div", {
                                          className: "publication-meta",
                                          children: [
                                            e.jsx("span", {
                                              className: "venue-badge",
                                              children: a.venue,
                                            }),
                                            e.jsx("time", { children: a.year }),
                                          ],
                                        }),
                                        e.jsx("h4", { children: a.title }),
                                        e.jsx("p", {
                                          className: "publication-authors",
                                          children: e.jsx(b, {
                                            text: a.authors,
                                          }),
                                        }),
                                        e.jsx("p", {
                                          className: "publication-venue",
                                          children: a.publication,
                                        }),
                                        e.jsxs("div", {
                                          className: "publication-links",
                                          children: [
                                            a.webpage &&
                                              e.jsxs(c, {
                                                href: a.webpage,
                                                children: [
                                                  "Web ",
                                                  e.jsx("span", {
                                                    "aria-hidden": "true",
                                                    children: "↗",
                                                  }),
                                                ],
                                              }),
                                            e.jsxs(c, {
                                              href: a.download,
                                              children: [
                                                "PDF ",
                                                e.jsx("span", {
                                                  "aria-hidden": "true",
                                                  children: "↓",
                                                }),
                                              ],
                                            }),
                                          ],
                                        }),
                                      ],
                                    }),
                                  ],
                                },
                                `${a.year}-${a.title}`,
                              ),
                            ),
                          }),
                        ],
                      },
                      s,
                    ),
                  ),
                ],
              }),
              e.jsxs("section", {
                className: "content-section",
                "aria-labelledby": "outputs",
                children: [
                  e.jsx(j, { id: "outputs", children: "Other Outputs" }),
                  e.jsxs("div", {
                    className: "output-columns",
                    children: [
                      e.jsxs("div", {
                        className: "output-block",
                        children: [
                          e.jsx("h3", { children: "Books" }),
                          e.jsx("ol", {
                            children: $.map((s) =>
                              e.jsxs(
                                "li",
                                {
                                  children: [
                                    e.jsx("strong", { children: s.title }),
                                    e.jsx("span", {
                                      children: e.jsx(b, { text: s.meta }),
                                    }),
                                    s.url &&
                                      e.jsx(c, {
                                        href: s.url,
                                        children:
                                          s.linkLabel ?? "View material ↗",
                                      }),
                                  ],
                                },
                                s.title,
                              ),
                            ),
                          }),
                        ],
                      }),
                      e.jsxs("div", {
                        className: "output-block",
                        children: [
                          e.jsx("h3", { children: "Patents" }),
                          e.jsx("ol", {
                            children: F.map((s) =>
                              e.jsxs(
                                "li",
                                {
                                  children: [
                                    e.jsx("strong", { children: s.title }),
                                    e.jsx("span", {
                                      children: e.jsx(b, { text: s.meta }),
                                    }),
                                    s.url &&
                                      e.jsx(c, {
                                        href: s.url,
                                        children:
                                          s.linkLabel ?? "View material ↗",
                                      }),
                                  ],
                                },
                                s.title,
                              ),
                            ),
                          }),
                        ],
                      }),
                      e.jsxs("div", {
                        className: "output-block",
                        children: [
                          e.jsx("h3", { children: "Software Copyrights" }),
                          e.jsx("ol", {
                            children: L.map((s) =>
                              e.jsxs(
                                "li",
                                {
                                  children: [
                                    e.jsx("strong", { children: s.title }),
                                    e.jsx("span", {
                                      children: e.jsx(b, { text: s.meta }),
                                    }),
                                    s.url &&
                                      e.jsx(c, {
                                        href: s.url,
                                        children:
                                          s.linkLabel ?? "View material ↗",
                                      }),
                                  ],
                                },
                                s.title,
                              ),
                            ),
                          }),
                        ],
                      }),
                    ],
                  }),
                ],
              }),
              e.jsxs("section", {
                className: "content-section",
                "aria-labelledby": "honors",
                children: [
                  e.jsx(j, { id: "honors", children: "Honors & Awards" }),
                  e.jsx("div", {
                    className: "simple-list",
                    children: O.map((s) =>
                      e.jsxs(
                        "div",
                        {
                          children: [
                            e.jsx("time", { children: s.year }),
                            e.jsx("p", { children: s.title }),
                            s.award && e.jsx("strong", { children: s.award }),
                          ],
                        },
                        `${s.year}-${s.title}`,
                      ),
                    ),
                  }),
                ],
              }),
              e.jsxs("section", {
                className: "content-section",
                "aria-labelledby": "experience",
                children: [
                  e.jsx(j, {
                    id: "experience",
                    children: "Professional Experience",
                  }),
                  e.jsx("div", {
                    className: "experience-list",
                    children: V.map((s) =>
                      e.jsxs(
                        "article",
                        {
                          children: [
                            e.jsx("time", { children: s.period }),
                            e.jsxs("div", {
                              children: [
                                e.jsx("h3", { children: s.organization }),
                                e.jsx("p", { children: s.role }),
                              ],
                            }),
                          ],
                        },
                        `${s.period}-${s.organization}`,
                      ),
                    ),
                  }),
                ],
              }),
              e.jsxs("section", {
                className: "content-section visitor-section",
                "aria-labelledby": "visitors",
                children: [
                  e.jsx(j, {
                    id: "visitors",
                    children: "Visitor Distribution",
                  }),
                  e.jsx(T, {}),
                ],
              }),
            ],
          }),
        ],
      }),
      e.jsxs("footer", {
        className: "site-footer",
        children: [
          e.jsx("p", { children: "© 2026 Yuang Wei" }),
          e.jsx("p", { children: "Last updated: August 2026" }),
        ],
      }),
    ],
  });
}
C.createRoot(document.getElementById("root")).render(
  e.jsx(u.StrictMode, { children: e.jsx(_, {}) }),
);
