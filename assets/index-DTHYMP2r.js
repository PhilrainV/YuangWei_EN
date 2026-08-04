import { r as h, j as e, c as C } from "./react-vendor-BTWVIjLd.js";
import { w as N } from "./world-map-CPvcksDd.js";
import {
  F as E,
  a as S,
  b as A,
  c as I,
  d as P,
  e as b,
  S as U,
} from "./icons-vendor-CUxSbwz5.js";
(function () {
  const o = document.createElement("link").relList;
  if (o && o.supports && o.supports("modulepreload")) return;
  for (const c of document.querySelectorAll('link[rel="modulepreload"]')) x(c);
  new MutationObserver((c) => {
    for (const d of c)
      if (d.type === "childList")
        for (const i of d.addedNodes)
          i.tagName === "LINK" && i.rel === "modulepreload" && x(i);
  }).observe(document, { childList: !0, subtree: !0 });
  function l(c) {
    const d = {};
    return (
      c.integrity && (d.integrity = c.integrity),
      c.referrerPolicy && (d.referrerPolicy = c.referrerPolicy),
      c.crossOrigin === "use-credentials"
        ? (d.credentials = "include")
        : c.crossOrigin === "anonymous"
          ? (d.credentials = "omit")
          : (d.credentials = "same-origin"),
      d
    );
  }
  function x(c) {
    if (c.ep) return;
    c.ep = !0;
    const d = l(c);
    fetch(c.href, d);
  }
})();
const j = window.YUANG_WEI_CONTENT ?? {};
function g(n) {
  return Array.isArray(n) ? n : [];
}
const M = g(j.publications),
  k = g(j.books),
  L = g(j.patents),
  F = g(j.softwareCopyrights),
  $ = g(j.honors),
  O = g(j.experience),
  y = "https://scholar.google.com/citations?user=jjXw5-4AAAAJ&hl=en",
  w = "https://yuang-wei-academic.philrain-cs.chatgpt.site/api/visitors",
  W =
    "https://raw.githubusercontent.com/PhilrainV/YuangWei_EN/google-scholar-stats/gs_data.json",
  V = [
    ["About", "about"],
    ["Education", "education"],
    ["Publications", "publications"],
    ["Other Outputs", "outputs"],
    ["Honors", "honors"],
    ["Experience", "experience"],
  ],
  Y = [
    "AI in Education",
    "Explainable AI",
    "Causal Models",
    "Knowledge Tracing",
    "Cognitive Diagnosis",
    "Large Language Models",
  ],
  T = new Intl.DisplayNames(["en"], { type: "region" });
function B(n) {
  const o =
    n.countryCode && n.countryCode !== "XX"
      ? T.of(n.countryCode)
      : "Unknown location";
  return [n.city, o].filter(Boolean).join(", ");
}
function f({ text: n }) {
  const o = n.split(/(Yuang Wei|Yu’ang Wei|Wei Yuang|魏雨昂)/g);
  return e.jsx(e.Fragment, {
    children: o.map((l, x) =>
      /^(Yuang Wei|Yu’ang Wei|Wei Yuang|魏雨昂)$/.test(l)
        ? e.jsx(
            "strong",
            { className: "author-self", children: l },
            `${l}-${x}`,
          )
        : e.jsx("span", { children: l }, `${l}-${x}`),
    ),
  });
}
function p({ id: n, children: o }) {
  return e.jsx("h2", { className: "section-heading", id: n, children: o });
}
function r({ href: n, children: o, className: l = "" }) {
  return e.jsx("a", {
    className: l,
    href: n,
    target: "_blank",
    rel: "noreferrer",
    children: o,
  });
}
function D() {
  const [n, o] = h.useState(null),
    [l, x] = h.useState(!1);
  h.useEffect(() => {
    let i = !0;
    async function u() {
      try {
        await fetch(w, { method: "POST", mode: "cors" });
        const m = await fetch(w, { cache: "no-store", mode: "cors" });
        if (!m.ok) throw new Error("visitor statistics unavailable");
        const v = await m.json();
        i && o(v);
      } catch {
        i && x(!0);
      }
    }
    return (
      u(),
      () => {
        i = !1;
      }
    );
  }, []);
  const c = h.useMemo(
      () => new Map((n?.countries ?? []).map((i) => [i.code.toLowerCase(), i])),
      [n],
    ),
    d = Math.max(1, ...(n?.countries ?? []).map((i) => i.visits));
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
              const u = c.get(i.id),
                m = u ? 0.3 + (u.visits / d) * 0.7 : 0;
              return e.jsx(
                "path",
                {
                  className: u ? "country-shape has-visits" : "country-shape",
                  d: i.path,
                  style: u ? { opacity: m } : void 0,
                  children: e.jsx("title", {
                    children: u ? `${i.name}: ${u.visits} visits` : i.name,
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
                          e.jsx("span", { children: B(i) }),
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
  const [n, o] = h.useState(!1),
    [l, x] = h.useState("All"),
    [c, d] = h.useState(!1),
    [i, u] = h.useState({
      citedby: 330,
      hindex: 11,
      i10index: 12,
      updated: "2026-08-02",
    });
  h.useEffect(() => {
    let s = !0;
    return (
      fetch(W, { cache: "no-store" })
        .then((a) => {
          if (!a.ok) throw new Error("scholar statistics unavailable");
          return a.json();
        })
        .then((a) => {
          s &&
            u((t) => ({
              citedby: Number(a.citedby ?? t.citedby),
              hindex: Number(a.hindex ?? t.hindex),
              i10index: Number(a.i10index ?? t.i10index),
              updated: a.updated ?? t.updated,
            }));
        })
        .catch(() => {}),
      () => {
        s = !1;
      }
    );
  }, []);
  const m = h.useMemo(
    () =>
      [
        "Journal Articles",
        "Chinese-Language Journal Articles",
        "Conference Papers",
      ]
        .filter(
          (a) =>
            l === "All" ||
            (l === "Journals"
              ? a !== "Conference Papers"
              : a === "Conference Papers"),
        )
        .map((a) => ({ group: a, papers: M.filter((t) => t.group === a) })),
    [l],
  );
  async function v() {
    const s = "philrain@foxmail.com";
    try {
      await navigator.clipboard.writeText(s);
    } catch {
      const a = document.createElement("textarea");
      ((a.value = s),
        (a.style.position = "fixed"),
        (a.style.opacity = "0"),
        document.body.appendChild(a),
        a.select(),
        document.execCommand("copy"),
        a.remove());
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
              children: "Yuang Wei",
            }),
            e.jsxs("button", {
              className: "menu-button",
              type: "button",
              "aria-label": n ? "Close navigation" : "Open navigation",
              "aria-expanded": n,
              onClick: () => o((s) => !s),
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
                V.map(([s, a]) =>
                  e.jsx(
                    "a",
                    { href: `#${a}`, onClick: () => o(!1), children: s },
                    a,
                  ),
                ),
                e.jsxs("a", {
                  className: "language-switch",
                  href: "https://philrainv.github.io/",
                  target: "_self",
                  "aria-label": "Switch to the Chinese homepage",
                  children: [e.jsx(E, { "aria-hidden": "true" }), "中文"],
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
                src: "images/weiyuang.png",
                alt: "Portrait of Yuang Wei",
              }),
              e.jsxs("div", {
                className: "profile-intro",
                children: [
                  e.jsx("h1", { children: "Yuang Wei" }),
                  e.jsx("p", {
                    className: "position",
                    children:
                      "Faculty of Artificial Intelligence in Education, Central China Normal University · Lecturer",
                  }),
                ],
              }),
              e.jsxs("div", {
                className: "contact-list",
                children: [
                  e.jsxs("div", {
                    className: "contact-location",
                    children: [
                      e.jsx(S, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "Wuhan, China",
                    ],
                  }),
                  e.jsxs("button", {
                    type: "button",
                    onClick: v,
                    children: [
                      e.jsx(A, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      c ? "Email copied" : "philrain@foxmail.com",
                    ],
                  }),
                  e.jsxs(r, {
                    href: "https://www.researchgate.net/profile/Yuang-Wei",
                    children: [
                      e.jsx(I, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "ResearchGate",
                    ],
                  }),
                  e.jsxs(r, {
                    href: "https://github.com/PhilrainV",
                    children: [
                      e.jsx(P, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "GitHub",
                    ],
                  }),
                  e.jsxs(r, {
                    href: y,
                    children: [
                      e.jsx(b, {
                        className: "contact-icon scholar-mark",
                        "aria-hidden": "true",
                      }),
                      "Google Scholar",
                    ],
                  }),
                  e.jsxs(r, {
                    href: "https://orcid.org/0000-0002-8187-4011",
                    children: [
                      e.jsx(U, {
                        className: "contact-icon",
                        "aria-hidden": "true",
                      }),
                      "ORCID",
                    ],
                  }),
                ],
              }),
              e.jsxs(r, {
                href: y,
                className: "scholar-card",
                children: [
                  e.jsxs("div", {
                    className: "scholar-card-title",
                    children: [
                      e.jsxs("span", {
                        children: [
                          e.jsx(b, { "aria-hidden": "true" }),
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
                  e.jsx(p, { id: "about", children: "About Me" }),
                  e.jsxs("div", {
                    className: "intro-text",
                    children: [
                      e.jsxs("p", {
                        children: [
                          "I received my Ph.D. in Intelligent Education from the",
                          " ",
                          e.jsx("strong", {
                            children:
                              "Shanghai Institute of AI for Education, East China Normal University (ECNU)",
                          }),
                          ", under the supervision of",
                          " ",
                          e.jsx(r, {
                            href: "https://faculty.ecnu.edu.cn/_s8/jb2/main.psp",
                            children: "Professor Bo Jiang",
                          }),
                          ". I am currently a Lecturer at the Faculty of Artificial Intelligence in Education, Central China Normal University (CCNU). I conduct research on trustworthy and explainable AI for education and have published more than 20 academic papers, including collaborative work.",
                        ],
                      }),
                      e.jsxs("p", {
                        children: [
                          "I serve as a reviewer for ",
                          e.jsx("em", { children: "Computers & Education" }),
                          ",",
                          " ",
                          e.jsx("em", {
                            children: "Education and Information Technologies",
                          }),
                          ",",
                          " ",
                          e.jsx("em", {
                            children: "Information Processing & Management",
                          }),
                          ",",
                          " ",
                          e.jsx("em", {
                            children:
                              "IEEE Transactions on Emerging Topics in Computing",
                          }),
                          ",",
                          " ",
                          e.jsx("em", {
                            children:
                              "International Journal of Artificial Intelligence in Education",
                          }),
                          ", ",
                          e.jsx("em", { children: "Knowledge-Based Systems" }),
                          ", and",
                          " ",
                          e.jsx("em", {
                            children:
                              "Humanities & Social Sciences Communications",
                          }),
                          ", as well as NeurIPS, AAAI, KDD, ICASSP, AIED, and EDM.",
                        ],
                      }),
                      e.jsx("p", {
                        children:
                          "If you are interested in my research, please feel free to contact me. I am always happy to discuss ideas and explore research collaborations.",
                      }),
                    ],
                  }),
                  e.jsxs("div", {
                    className: "research-row",
                    children: [
                      e.jsx("strong", { children: "Research" }),
                      e.jsx("div", {
                        children: Y.map((s) =>
                          e.jsx("span", { children: s }, s),
                        ),
                      }),
                    ],
                  }),
                ],
              }),
              e.jsxs("section", {
                className: "content-section",
                "aria-labelledby": "education",
                children: [
                  e.jsx(p, { id: "education", children: "Education" }),
                  e.jsxs("div", {
                    className: "education-list",
                    children: [
                      e.jsxs("article", {
                        className: "education-item",
                        children: [
                          e.jsx(r, {
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
                                children: e.jsx(r, {
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
                                  e.jsx(r, {
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
                          e.jsx(r, {
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
                                children: e.jsx(r, {
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
                                  e.jsx(r, {
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
                          e.jsx(r, {
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
                                children: e.jsx(r, {
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
                                  e.jsx(r, {
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
                      e.jsx(p, {
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
                              onClick: () => x(s),
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
                  m.map(({ group: s, papers: a }) =>
                    e.jsxs(
                      "div",
                      {
                        className: "publication-group",
                        children: [
                          e.jsxs("h3", {
                            className: "publication-group-title",
                            children: [
                              s,
                              e.jsx("span", { children: a.length }),
                            ],
                          }),
                          e.jsx("div", {
                            className: "publication-list",
                            children: a.map((t) =>
                              e.jsxs(
                                "article",
                                {
                                  className: t.image
                                    ? "publication-item with-image"
                                    : "publication-item",
                                  children: [
                                    t.image &&
                                      e.jsx("img", {
                                        className: "publication-image",
                                        src: t.image,
                                        alt: `Illustration for ${t.title}`,
                                        loading: "lazy",
                                        style: {
                                          objectFit: t.imageFit || "cover",
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
                                              children: t.venue,
                                            }),
                                            e.jsx("time", { children: t.year }),
                                          ],
                                        }),
                                        e.jsx("h4", { children: t.title }),
                                        e.jsx("p", {
                                          className: "publication-authors",
                                          children: e.jsx(f, {
                                            text: t.authors,
                                          }),
                                        }),
                                        e.jsx("p", {
                                          className: "publication-venue",
                                          children: t.publication,
                                        }),
                                        e.jsxs("div", {
                                          className: "publication-links",
                                          children: [
                                            t.webpage &&
                                              e.jsxs(r, {
                                                href: t.webpage,
                                                children: [
                                                  "Web ",
                                                  e.jsx("span", {
                                                    "aria-hidden": "true",
                                                    children: "↗",
                                                  }),
                                                ],
                                              }),
                                            e.jsxs(r, {
                                              href: t.download,
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
                                `${t.year}-${t.title}`,
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
                  e.jsx(p, { id: "outputs", children: "Other Outputs" }),
                  e.jsxs("div", {
                    className: "output-columns",
                    children: [
                      e.jsxs("div", {
                        className: "output-block",
                        children: [
                          e.jsx("h3", { children: "Books" }),
                          e.jsx("ol", {
                            children: k.map((s) =>
                              e.jsxs(
                                "li",
                                {
                                  children: [
                                    e.jsx("strong", { children: s.title }),
                                    e.jsx("span", {
                                      children: e.jsx(f, { text: s.meta }),
                                    }),
                                    s.url &&
                                      e.jsx(r, {
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
                                      e.jsx(r, {
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
                            children: F.map((s) =>
                              e.jsxs(
                                "li",
                                {
                                  children: [
                                    e.jsx("strong", { children: s.title }),
                                    e.jsx("span", {
                                      children: e.jsx(f, { text: s.meta }),
                                    }),
                                    s.url &&
                                      e.jsx(r, {
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
                  e.jsx(p, { id: "honors", children: "Honors & Awards" }),
                  e.jsx("div", {
                    className: "simple-list",
                    children: $.map((s) =>
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
                  e.jsx(p, {
                    id: "experience",
                    children: "Professional Experience",
                  }),
                  e.jsx("div", {
                    className: "experience-list",
                    children: O.map((s) =>
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
                  e.jsx(p, {
                    id: "visitors",
                    children: "Visitor Distribution",
                  }),
                  e.jsx(D, {}),
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
  e.jsx(h.StrictMode, { children: e.jsx(_, {}) }),
);
