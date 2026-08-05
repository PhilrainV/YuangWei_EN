import { r as u, j as e, c as C } from "./react-vendor-BTWVIjLd.js";
import { w as N } from "./world-map-CPvcksDd.js";
import {
  F as S,
  a as k,
  b as E,
  c as A,
  d as $,
  e as y,
  S as F,
} from "./icons-vendor-CUxSbwz5.js";
(function () {
  const o = document.createElement("link").relList;
  if (o && o.supports && o.supports("modulepreload")) return;
  for (const t of document.querySelectorAll('link[rel="modulepreload"]')) p(t);
  new MutationObserver((t) => {
    for (const d of t)
      if (d.type === "childList")
        for (const i of d.addedNodes)
          i.tagName === "LINK" && i.rel === "modulepreload" && p(i);
  }).observe(document, { childList: !0, subtree: !0 });
  function l(t) {
    const d = {};
    return (
      t.integrity && (d.integrity = t.integrity),
      t.referrerPolicy && (d.referrerPolicy = t.referrerPolicy),
      t.crossOrigin === "use-credentials"
        ? (d.credentials = "include")
        : t.crossOrigin === "anonymous"
          ? (d.credentials = "omit")
          : (d.credentials = "same-origin"),
      d
    );
  }
  function p(t) {
    if (t.ep) return;
    t.ep = !0;
    const d = l(t);
    fetch(t.href, d);
  }
})();
const j = window.YUANG_WEI_CONTENT ?? {};
function g(a) {
  return Array.isArray(a) ? a : [];
}
const P = g(j.publications),
  c = j.profile,
  M = g(j.education),
  O = g(j.books),
  L = g(j.patents),
  V = g(j.softwareCopyrights),
  W = g(j.honors),
  Y = g(j.experience),
  w = "https://yuang-wei-academic.philrain-cs.chatgpt.site/api/visitors",
  U =
    "https://raw.githubusercontent.com/PhilrainV/YuangWei_EN/google-scholar-stats/gs_data.json",
  G = [
    ["About", "about"],
    ["Education", "education"],
    ["Publications", "publications"],
    ["Other Outputs", "outputs"],
    ["Honors", "honors"],
    ["Experience", "experience"],
  ],
  B = new Intl.DisplayNames(["en"], { type: "region" });
function I(a) {
  const o =
    a.countryCode && a.countryCode !== "XX"
      ? B.of(a.countryCode)
      : "Unknown location";
  return [a.city, o].filter(Boolean).join(", ");
}
function f({ text: a }) {
  const o = a.split(/(Yuang Wei|Yu’ang Wei|Wei Yuang|魏雨昂)/g);
  return e.jsx(e.Fragment, {
    children: o.map((l, p) =>
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
function m({ id: a, children: o }) {
  return e.jsx("h2", { className: "section-heading", id: a, children: o });
}
function h({ href: a, children: o, className: l = "" }) {
  return e.jsx("a", {
    className: l,
    href: a,
    target: "_blank",
    rel: "noreferrer",
    children: o,
  });
}
function R() {
  const [a, o] = u.useState(null),
    [l, p] = u.useState(!1);
  u.useEffect(() => {
    let i = !0;
    async function x() {
      try {
        await fetch(w, { method: "POST", mode: "cors" });
        const b = await fetch(w, { cache: "no-store", mode: "cors" });
        if (!b.ok) throw new Error("visitor statistics unavailable");
        const v = await b.json();
        i && o(v);
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
      () => new Map((a?.countries ?? []).map((i) => [i.code.toLowerCase(), i])),
      [a],
    ),
    d = Math.max(1, ...(a?.countries ?? []).map((i) => i.visits));
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
                b = x ? 0.3 + (x.visits / d) * 0.7 : 0;
              return e.jsx(
                "path",
                {
                  className: x ? "country-shape has-visits" : "country-shape",
                  d: i.path,
                  style: x ? { opacity: b } : void 0,
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
                  e.jsx("strong", { children: a?.totalVisits ?? "—" }),
                  e.jsx("span", { children: "Total visits" }),
                ],
              }),
              e.jsxs("div", {
                children: [
                  e.jsx("strong", { children: a?.uniqueVisitors ?? "—" }),
                  e.jsx("span", { children: "Unique visitors" }),
                ],
              }),
              e.jsxs("div", {
                children: [
                  e.jsx("strong", { children: a?.countries.length ?? "—" }),
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
            : a?.recentVisitors.length
              ? e.jsx("ol", {
                  className: "recent-visitors",
                  children: a.recentVisitors.map((i) =>
                    e.jsxs(
                      "li",
                      {
                        children: [
                          e.jsxs("span", {
                            className: "visitor-id",
                            children: ["Visitor ", i.id],
                          }),
                          e.jsx("span", { children: I(i) }),
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
function T() {
  const [a, o] = u.useState(!1),
    [l, p] = u.useState("All"),
    [t, d] = u.useState(!1),
    [i, x] = u.useState({
      citedby: 330,
      hindex: 11,
      i10index: 12,
      updated: "2026-08-02",
    });
  u.useEffect(() => {
    let s = !0;
    return (
      fetch(U, { cache: "no-store" })
        .then((r) => {
          if (!r.ok) throw new Error("scholar statistics unavailable");
          return r.json();
        })
        .then((r) => {
          s &&
            x((n) => ({
              citedby: Number(r.citedby ?? n.citedby),
              hindex: Number(r.hindex ?? n.hindex),
              i10index: Number(r.i10index ?? n.i10index),
              updated: r.updated ?? n.updated,
            }));
        })
        .catch(() => {}),
      () => {
        s = !1;
      }
    );
  }, []);
  const b = u.useMemo(
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
        .map((r) => ({ group: r, papers: P.filter((n) => n.group === r) })),
    [l],
  );
  async function v() {
    try {
      await navigator.clipboard.writeText(c.email);
    } catch {
      const s = document.createElement("textarea");
      ((s.value = c.email),
        (s.style.position = "fixed"),
        (s.style.opacity = "0"),
        document.body.appendChild(s),
        s.select(),
        document.execCommand("copy"),
        s.remove());
    }
    (d(!0), window.setTimeout(() => d(!1), 1600));
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
              children: c.name,
            }),
            e.jsxs("button", {
              className: "menu-button",
              type: "button",
              "aria-label": a ? "Close navigation" : "Open navigation",
              "aria-expanded": a,
              onClick: () => o((s) => !s),
              children: [
                e.jsx("span", {}),
                e.jsx("span", {}),
                e.jsx("span", {}),
              ],
            }),
            e.jsxs("nav", {
              className: a ? "main-nav is-open" : "main-nav",
              "aria-label": "Main navigation",
              children: [
                G.map(([s, r]) =>
                  e.jsx(
                    "a",
                    { href: `#${r}`, onClick: () => o(!1), children: s },
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
                src: c.avatar,
                alt: c.avatarAlt,
              }),
              e.jsxs("div", {
                className: "profile-intro",
                children: [
                  e.jsx("h1", { children: c.name }),
                  e.jsxs("p", {
                    className: "position",
                    children: [c.affiliation, " · ", c.title],
                  }),
                ],
              }),
              e.jsxs("div", {
                className: "contact-list",
                children: [
                  e.jsxs("div", {
                    className: "contact-location",
                    children: [
                      e.jsx(k, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      c.location,
                    ],
                  }),
                  e.jsxs("button", {
                    type: "button",
                    onClick: v,
                    children: [
                      e.jsx(E, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      t ? "Email copied" : c.email,
                    ],
                  }),
                  e.jsxs(h, {
                    href: c.links.researchGate,
                    children: [
                      e.jsx(A, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "ResearchGate",
                    ],
                  }),
                  e.jsxs(h, {
                    href: c.links.github,
                    children: [
                      e.jsx($, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "GitHub",
                    ],
                  }),
                  e.jsxs(h, {
                    href: c.links.googleScholar,
                    children: [
                      e.jsx(y, {
                        className: "contact-icon scholar-mark",
                        "aria-hidden": "true",
                      }),
                      "Google Scholar",
                    ],
                  }),
                  e.jsxs(h, {
                    href: c.links.orcid,
                    children: [
                      e.jsx(F, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "ORCID",
                    ],
                  }),
                ],
              }),
              e.jsxs(h, {
                href: c.links.googleScholar,
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
                  e.jsx(m, { id: "about", children: "About Me" }),
                  e.jsx("div", {
                    className: "intro-text",
                    children: c.bio.map((s) => e.jsx("p", { children: s }, s)),
                  }),
                  e.jsxs("div", {
                    className: "research-row",
                    children: [
                      e.jsx("strong", { children: "Research" }),
                      e.jsx("div", {
                        children: c.researchInterests.map((s) =>
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
                  e.jsx(m, { id: "education", children: "Education" }),
                  e.jsx("div", {
                    className: "education-list",
                    children: M.map((s) =>
                      e.jsxs(
                        "article",
                        {
                          className: "education-item",
                          children: [
                            e.jsx(h, {
                              href: s.institutionUrl,
                              className: "school-logo-link",
                              children: e.jsx("img", {
                                src: s.logo,
                                alt: s.logoAlt,
                              }),
                            }),
                            e.jsx("div", {
                              className: "education-time",
                              children: s.period,
                            }),
                            e.jsxs("div", {
                              className: "education-body",
                              children: [
                                e.jsx("h3", {
                                  children: e.jsx(h, {
                                    href: s.institutionUrl,
                                    children: s.institution,
                                  }),
                                }),
                                e.jsx("p", { children: s.program }),
                                e.jsxs("p", {
                                  className: "education-note",
                                  children: [
                                    s.supervisor &&
                                      e.jsxs(e.Fragment, {
                                        children: [
                                          "Supervisor:",
                                          " ",
                                          s.supervisorUrl
                                            ? e.jsx(h, {
                                                href: s.supervisorUrl,
                                                children: s.supervisor,
                                              })
                                            : s.supervisor,
                                        ],
                                      }),
                                    e.jsx("span", { children: s.location }),
                                  ],
                                }),
                              ],
                            }),
                          ],
                        },
                        `${s.period}-${s.institution}`,
                      ),
                    ),
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
                      e.jsx(m, {
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
                  b.map(({ group: s, papers: r }) =>
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
                            children: r.map((n) =>
                              e.jsxs(
                                "article",
                                {
                                  className: n.image
                                    ? "publication-item with-image"
                                    : "publication-item",
                                  children: [
                                    n.image &&
                                      e.jsx("img", {
                                        className: "publication-image",
                                        src: n.image,
                                        alt: `Illustration for ${n.title}`,
                                        loading: "lazy",
                                        style: {
                                          objectFit: n.imageFit || "cover",
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
                                              children: n.venue,
                                            }),
                                            e.jsx("time", { children: n.year }),
                                          ],
                                        }),
                                        e.jsx("h4", { children: n.title }),
                                        e.jsx("p", {
                                          className: "publication-authors",
                                          children: e.jsx(f, {
                                            text: n.authors,
                                          }),
                                        }),
                                        e.jsx("p", {
                                          className: "publication-venue",
                                          children: n.publication,
                                        }),
                                        e.jsxs("div", {
                                          className: "publication-links",
                                          children: [
                                            n.webpage &&
                                              e.jsxs(h, {
                                                href: n.webpage,
                                                children: [
                                                  "Web ",
                                                  e.jsx("span", {
                                                    "aria-hidden": "true",
                                                    children: "↗",
                                                  }),
                                                ],
                                              }),
                                            e.jsxs(h, {
                                              href: n.download,
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
                                `${n.year}-${n.title}`,
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
                  e.jsx(m, { id: "outputs", children: "Other Outputs" }),
                  e.jsxs("div", {
                    className: "output-columns",
                    children: [
                      e.jsxs("div", {
                        className: "output-block",
                        children: [
                          e.jsx("h3", { children: "Books" }),
                          e.jsx("ol", {
                            children: O.map((s) =>
                              e.jsxs(
                                "li",
                                {
                                  children: [
                                    e.jsx("strong", { children: s.title }),
                                    e.jsx("span", {
                                      children: e.jsx(f, { text: s.meta }),
                                    }),
                                    s.url &&
                                      e.jsx(h, {
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
                            children: L.map((s) =>
                              e.jsxs(
                                "li",
                                {
                                  children: [
                                    e.jsx("strong", { children: s.title }),
                                    e.jsx("span", {
                                      children: e.jsx(f, { text: s.meta }),
                                    }),
                                    s.url &&
                                      e.jsx(h, {
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
                            children: V.map((s) =>
                              e.jsxs(
                                "li",
                                {
                                  children: [
                                    e.jsx("strong", { children: s.title }),
                                    e.jsx("span", {
                                      children: e.jsx(f, { text: s.meta }),
                                    }),
                                    s.url &&
                                      e.jsx(h, {
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
                  e.jsx(m, { id: "honors", children: "Honors & Awards" }),
                  e.jsx("div", {
                    className: "simple-list",
                    children: W.map((s) =>
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
                  e.jsx(m, {
                    id: "experience",
                    children: "Professional Experience",
                  }),
                  e.jsx("div", {
                    className: "experience-list",
                    children: Y.map((s) =>
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
                  e.jsx(m, {
                    id: "visitors",
                    children: "Visitor Distribution",
                  }),
                  e.jsx(R, {}),
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
  e.jsx(u.StrictMode, { children: e.jsx(T, {}) }),
);
