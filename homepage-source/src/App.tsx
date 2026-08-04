"use client";

import worldMap from "@svg-maps/world";
import { useEffect, useMemo, useState } from "react";
import {
  FaEnvelope,
  FaGithub,
  FaGraduationCap,
  FaLanguage,
  FaMapMarkerAlt,
  FaResearchgate,
} from "react-icons/fa";
import { SiOrcid } from "react-icons/si";
import {
  books,
  experience,
  honors,
  patents,
  profile,
  publications,
  softwareCopyrights,
  type PublicationGroup,
} from "./content";

const visitorApiUrl =
  "https://yuang-wei-academic.philrain-cs.chatgpt.site/api/visitors";
const scholarStatsUrl =
  "https://raw.githubusercontent.com/PhilrainV/YuangWei_EN/google-scholar-stats/gs_data.json";

const navigation = [
  ["About", "about"],
  ["Education", "education"],
  ["Publications", "publications"],
  ["Other Outputs", "outputs"],
  ["Honors", "honors"],
  ["Experience", "experience"],
] as const;

type VisitorCountry = {
  code: string;
  visits: number;
  visitors: number;
};

type RecentVisitor = {
  id: string;
  countryCode: string;
  country: string;
  city: string;
  lastSeen: string;
};

type VisitorStats = {
  totalVisits: number;
  uniqueVisitors: number;
  countries: VisitorCountry[];
  recentVisitors: RecentVisitor[];
};

type ScholarStats = {
  citedby: number;
  hindex: number;
  i10index: number;
  updated?: string;
};

const englishRegionNames = new Intl.DisplayNames(["en"], { type: "region" });

function visitorLocation(visitor: RecentVisitor) {
  const country =
    visitor.countryCode && visitor.countryCode !== "XX"
      ? englishRegionNames.of(visitor.countryCode)
      : "Unknown location";
  return [visitor.city, country].filter(Boolean).join(", ");
}

function HighlightedAuthors({ text }: { text: string }) {
  const parts = text.split(/(Yuang Wei|Yu’ang Wei|Wei Yuang|魏雨昂)/g);
  return (
    <>
      {parts.map((part, index) =>
        /^(Yuang Wei|Yu’ang Wei|Wei Yuang|魏雨昂)$/.test(part) ? (
          <strong className="author-self" key={`${part}-${index}`}>
            {part}
          </strong>
        ) : (
          <span key={`${part}-${index}`}>{part}</span>
        ),
      )}
    </>
  );
}

function SectionHeading({
  id,
  children,
}: {
  id: string;
  children: React.ReactNode;
}) {
  return (
    <h2 className="section-heading" id={id}>
      {children}
    </h2>
  );
}

function ExternalLink({
  href,
  children,
  className = "",
}: {
  href: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <a className={className} href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

function VisitorMap() {
  const [stats, setStats] = useState<VisitorStats | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    let active = true;

    async function registerAndLoad() {
      try {
        await fetch(visitorApiUrl, { method: "POST", mode: "cors" });
        const response = await fetch(visitorApiUrl, {
          cache: "no-store",
          mode: "cors",
        });
        if (!response.ok) throw new Error("visitor statistics unavailable");
        const data = (await response.json()) as VisitorStats;
        if (active) setStats(data);
      } catch {
        if (active) setUnavailable(true);
      }
    }

    registerAndLoad();
    return () => {
      active = false;
    };
  }, []);

  const visitsByCountry = useMemo(
    () =>
      new Map(
        (stats?.countries ?? []).map((item) => [item.code.toLowerCase(), item]),
      ),
    [stats],
  );
  const maxVisits = Math.max(
    1,
    ...(stats?.countries ?? []).map((item) => item.visits),
  );

  return (
    <div className="visitor-dashboard">
      <div className="visitor-map-panel">
        <svg
          className="world-map"
          viewBox={worldMap.viewBox}
          role="img"
          aria-label="World map of website visitors"
        >
          {worldMap.locations.map(
            (location: { id: string; name: string; path: string }) => {
              const country = visitsByCountry.get(location.id);
              const intensity = country
                ? 0.3 + (country.visits / maxVisits) * 0.7
                : 0;
              return (
                <path
                  className={
                    country ? "country-shape has-visits" : "country-shape"
                  }
                  d={location.path}
                  key={location.id}
                  style={country ? { opacity: intensity } : undefined}
                >
                  <title>
                    {country
                      ? `${location.name}: ${country.visits} visits`
                      : location.name}
                  </title>
                </path>
              );
            },
          )}
        </svg>
        <p className="map-credit">Map data: SVG Maps (CC BY 4.0)</p>
      </div>

      <div className="visitor-summary" aria-live="polite">
        <div className="visitor-metrics">
          <div>
            <strong>{stats?.totalVisits ?? "—"}</strong>
            <span>Total visits</span>
          </div>
          <div>
            <strong>{stats?.uniqueVisitors ?? "—"}</strong>
            <span>Unique visitors</span>
          </div>
          <div>
            <strong>{stats?.countries.length ?? "—"}</strong>
            <span>Countries / regions</span>
          </div>
        </div>
        <h3>Recent visitors</h3>
        {unavailable ? (
          <p className="visitor-empty">
            Visitor statistics are temporarily unavailable.
          </p>
        ) : stats?.recentVisitors.length ? (
          <ol className="recent-visitors">
            {stats.recentVisitors.map((visitor) => (
              <li key={`${visitor.id}-${visitor.lastSeen}`}>
                <span className="visitor-id">Visitor {visitor.id}</span>
                <span>{visitorLocation(visitor)}</span>
              </li>
            ))}
          </ol>
        ) : (
          <p className="visitor-empty">Loading visitor data…</p>
        )}
        <p className="privacy-note">
          To protect visitor privacy, only an anonymous identifier and
          approximate location are displayed. Full IP addresses are never shown.
        </p>
      </div>
    </div>
  );
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [paperFilter, setPaperFilter] = useState<
    "All" | "Journals" | "Conferences"
  >("All");
  const [emailCopied, setEmailCopied] = useState(false);
  const [scholarMetrics, setScholarMetrics] = useState<ScholarStats>({
    citedby: 330,
    hindex: 11,
    i10index: 12,
    updated: "2026-08-02",
  });

  useEffect(() => {
    let active = true;

    fetch(scholarStatsUrl, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error("scholar statistics unavailable");
        return response.json() as Promise<Partial<ScholarStats>>;
      })
      .then((data) => {
        if (!active) return;
        setScholarMetrics((current) => ({
          citedby: Number(data.citedby ?? current.citedby),
          hindex: Number(data.hindex ?? current.hindex),
          i10index: Number(data.i10index ?? current.i10index),
          updated: data.updated ?? current.updated,
        }));
      })
      .catch(() => undefined);

    return () => {
      active = false;
    };
  }, []);

  const groupedPublications = useMemo(() => {
    const groups: PublicationGroup[] = [
      "Journal Articles",
      "Chinese-Language Journal Articles",
      "Conference Papers",
    ];
    return groups
      .filter(
        (group) =>
          paperFilter === "All" ||
          (paperFilter === "Journals"
            ? group !== "Conference Papers"
            : group === "Conference Papers"),
      )
      .map((group) => ({
        group,
        papers: publications.filter((paper) => paper.group === group),
      }));
  }, [paperFilter]);

  async function copyEmail() {
    try {
      await navigator.clipboard.writeText(profile.email);
    } catch {
      const helper = document.createElement("textarea");
      helper.value = profile.email;
      helper.style.position = "fixed";
      helper.style.opacity = "0";
      document.body.appendChild(helper);
      helper.select();
      document.execCommand("copy");
      helper.remove();
    }
    setEmailCopied(true);
    window.setTimeout(() => setEmailCopied(false), 1600);
  }

  return (
    <>
      <header className="site-header">
        <div className="header-inner">
          <a className="site-title" href="#about" aria-label="Back to the top">
            {profile.name}
          </a>
          <button
            className="menu-button"
            type="button"
            aria-label={menuOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((value) => !value)}
          >
            <span />
            <span />
            <span />
          </button>
          <nav
            className={menuOpen ? "main-nav is-open" : "main-nav"}
            aria-label="Main navigation"
          >
            {navigation.map(([label, id]) => (
              <a href={`#${id}`} key={id} onClick={() => setMenuOpen(false)}>
                {label}
              </a>
            ))}
            <a
              className="language-switch"
              href="https://philrainv.github.io/"
              target="_self"
              aria-label="Switch to the Chinese homepage"
            >
              <FaLanguage aria-hidden="true" />
              中文
            </a>
          </nav>
        </div>
      </header>

      <main className="page-shell">
        <aside className="profile-panel" aria-label="Profile">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            className="portrait"
            src={profile.avatar}
            alt={profile.avatarAlt}
          />
          <div className="profile-intro">
            <h1>{profile.name}</h1>
            <p className="position">
              {profile.affiliation} · {profile.title}
            </p>
          </div>

          <div className="contact-list">
            <div className="contact-location">
              <FaMapMarkerAlt className="contact-icon" aria-hidden="true" />
              {profile.location}
            </div>
            <button type="button" onClick={copyEmail}>
              <FaEnvelope className="contact-icon" aria-hidden="true" />
              {emailCopied ? "Email copied" : profile.email}
            </button>
            <ExternalLink href={profile.links.researchGate}>
              <FaResearchgate className="contact-icon" aria-hidden="true" />
              ResearchGate
            </ExternalLink>
            <ExternalLink href={profile.links.github}>
              <FaGithub className="contact-icon" aria-hidden="true" />
              GitHub
            </ExternalLink>
            <ExternalLink href={profile.links.googleScholar}>
              <FaGraduationCap
                className="contact-icon scholar-mark"
                aria-hidden="true"
              />
              Google Scholar
            </ExternalLink>
            <ExternalLink href={profile.links.orcid}>
              <SiOrcid className="contact-icon" aria-hidden="true" />
              ORCID
            </ExternalLink>
          </div>

          <ExternalLink
            href={profile.links.googleScholar}
            className="scholar-card"
          >
            <div className="scholar-card-title">
              <span>
                <FaGraduationCap aria-hidden="true" /> Google Scholar
              </span>
              <span aria-hidden="true">↗</span>
            </div>
            <div className="scholar-metrics">
              <div>
                <strong>{scholarMetrics.citedby}</strong>
                <span>Citations</span>
              </div>
              <div>
                <strong>{scholarMetrics.hindex}</strong>
                <span>h-index</span>
              </div>
              <div>
                <strong>{scholarMetrics.i10index}</strong>
                <span>i10-index</span>
              </div>
            </div>
            <small>
              Automatically updated
              {scholarMetrics.updated
                ? ` · ${scholarMetrics.updated.slice(0, 10).replaceAll("-", ".")}`
                : ""}
            </small>
          </ExternalLink>
        </aside>

        <div className="main-content">
          <section
            className="content-section about-section"
            aria-labelledby="about"
          >
            <SectionHeading id="about">About Me</SectionHeading>
            <div className="intro-text">
              {profile.bio.map((paragraph) => (
                <p key={paragraph}>{paragraph}</p>
              ))}
            </div>
            <div className="research-row">
              <strong>Research</strong>
              <div>
                {profile.researchInterests.map((interest) => (
                  <span key={interest}>{interest}</span>
                ))}
              </div>
            </div>
          </section>

          <section
            className="content-section education-section"
            aria-labelledby="education"
          >
            <SectionHeading id="education">Education</SectionHeading>
            <div className="education-list">
              <article className="education-item">
                <ExternalLink
                  href="https://aiedu.ecnu.edu.cn/"
                  className="school-logo-link"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src="images/ECNU_logo.png"
                    alt="East China Normal University logo"
                  />
                </ExternalLink>
                <div className="education-time">2022.06 — 2026.06</div>
                <div className="education-body">
                  <h3>
                    <ExternalLink href="https://aiedu.ecnu.edu.cn/">
                      East China Normal University (ECNU)
                    </ExternalLink>
                  </h3>
                  <p>
                    Shanghai Institute of AI for Education · Intelligent
                    Education · Ph.D.
                  </p>
                  <p className="education-note">
                    Supervisor:{" "}
                    <ExternalLink href="https://faculty.ecnu.edu.cn/_s8/jb2/main.psp">
                      Prof. Bo Jiang
                    </ExternalLink>
                    <span>Shanghai, China</span>
                  </p>
                </div>
              </article>
              <article className="education-item">
                <ExternalLink
                  href="https://www.comp.nus.edu.sg/cs/"
                  className="school-logo-link"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src="images/NUS_logo.png"
                    alt="National University of Singapore logo"
                  />
                </ExternalLink>
                <div className="education-time">2024.09 — 2025.09</div>
                <div className="education-body">
                  <h3>
                    <ExternalLink href="https://www.comp.nus.edu.sg/cs/">
                      National University of Singapore (NUS)
                    </ExternalLink>
                  </h3>
                  <p>
                    Department of Computer Science · Human–Computer Interaction
                    · CSC Visiting Ph.D. Student
                  </p>
                  <p className="education-note">
                    Supervisor:{" "}
                    <ExternalLink href="https://www.comp.nus.edu.sg/cs/people/brianlim/">
                      Assoc. Prof. Brian Y. Lim
                    </ExternalLink>
                    <span>Singapore</span>
                  </p>
                </div>
              </article>
              <article className="education-item">
                <ExternalLink
                  href="https://dqgc.ncut.edu.cn/"
                  className="school-logo-link"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src="images/NCUT_logo.png"
                    alt="North China University of Technology logo"
                  />
                </ExternalLink>
                <div className="education-time">2015.09 — 2022.06</div>
                <div className="education-body">
                  <h3>
                    <ExternalLink href="https://dqgc.ncut.edu.cn/">
                      North China University of Technology (NCUT)
                    </ExternalLink>
                  </h3>
                  <p>
                    School of Electrical and Control Engineering · B.Eng. in
                    Automation and M.Eng. in Control Science and Engineering
                  </p>
                  <p className="education-note">
                    Supervisor:{" "}
                    <ExternalLink href="https://dqgc.ncut.edu.cn/info/1228/3137.htm">
                      Assoc. Prof. Jining Xu
                    </ExternalLink>
                    <span>Beijing, China</span>
                  </p>
                </div>
              </article>
            </div>
          </section>

          <section className="content-section" aria-labelledby="publications">
            <div className="heading-with-tools">
              <SectionHeading id="publications">Publications</SectionHeading>
              <div
                className="paper-filter"
                aria-label="Filter publications by type"
              >
                {(["All", "Journals", "Conferences"] as const).map((item) => (
                  <button
                    type="button"
                    key={item}
                    className={paperFilter === item ? "is-active" : ""}
                    aria-pressed={paperFilter === item}
                    onClick={() => setPaperFilter(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <p className="publication-legend">
              <span>Yuang Wei</span> indicates my name; publications are grouped
              by type and ordered by year.
            </p>

            {groupedPublications.map(({ group, papers }) => (
              <div className="publication-group" key={group}>
                <h3 className="publication-group-title">
                  {group}
                  <span>{papers.length}</span>
                </h3>
                <div className="publication-list">
                  {papers.map((paper) => (
                    <article
                      className={
                        paper.image
                          ? "publication-item with-image"
                          : "publication-item"
                      }
                      key={`${paper.year}-${paper.title}`}
                    >
                      {paper.image && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          className="publication-image"
                          src={paper.image}
                          alt={`Illustration for ${paper.title}`}
                          loading="lazy"
                          style={{ objectFit: paper.imageFit || "cover" }}
                        />
                      )}
                      <div className="publication-main">
                        <div className="publication-meta">
                          <span className="venue-badge">{paper.venue}</span>
                          <time>{paper.year}</time>
                        </div>
                        <h4>{paper.title}</h4>
                        <p className="publication-authors">
                          <HighlightedAuthors text={paper.authors} />
                        </p>
                        <p className="publication-venue">{paper.publication}</p>
                        <div className="publication-links">
                          {paper.webpage && (
                            <ExternalLink href={paper.webpage}>
                              Web <span aria-hidden="true">↗</span>
                            </ExternalLink>
                          )}
                          <ExternalLink href={paper.download}>
                            PDF <span aria-hidden="true">↓</span>
                          </ExternalLink>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </div>
            ))}
          </section>

          <section className="content-section" aria-labelledby="outputs">
            <SectionHeading id="outputs">Other Outputs</SectionHeading>
            <div className="output-columns">
              <div className="output-block">
                <h3>Books</h3>
                <ol>
                  {books.map((item) => (
                    <li key={item.title}>
                      <strong>{item.title}</strong>
                      <span>
                        <HighlightedAuthors text={item.meta} />
                      </span>
                      {item.url && (
                        <ExternalLink href={item.url}>
                          {item.linkLabel ?? "View material ↗"}
                        </ExternalLink>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
              <div className="output-block">
                <h3>Patents</h3>
                <ol>
                  {patents.map((item) => (
                    <li key={item.title}>
                      <strong>{item.title}</strong>
                      <span>
                        <HighlightedAuthors text={item.meta} />
                      </span>
                      {item.url && (
                        <ExternalLink href={item.url}>
                          {item.linkLabel ?? "View material ↗"}
                        </ExternalLink>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
              <div className="output-block">
                <h3>Software Copyrights</h3>
                <ol>
                  {softwareCopyrights.map((item) => (
                    <li key={item.title}>
                      <strong>{item.title}</strong>
                      <span>
                        <HighlightedAuthors text={item.meta} />
                      </span>
                      {item.url && (
                        <ExternalLink href={item.url}>
                          {item.linkLabel ?? "View material ↗"}
                        </ExternalLink>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          </section>

          <section className="content-section" aria-labelledby="honors">
            <SectionHeading id="honors">Honors & Awards</SectionHeading>
            <div className="simple-list">
              {honors.map((item) => (
                <div key={`${item.year}-${item.title}`}>
                  <time>{item.year}</time>
                  <p>{item.title}</p>
                  {item.award && <strong>{item.award}</strong>}
                </div>
              ))}
            </div>
          </section>

          <section className="content-section" aria-labelledby="experience">
            <SectionHeading id="experience">
              Professional Experience
            </SectionHeading>
            <div className="experience-list">
              {experience.map((item) => (
                <article key={`${item.period}-${item.organization}`}>
                  <time>{item.period}</time>
                  <div>
                    <h3>{item.organization}</h3>
                    <p>{item.role}</p>
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section
            className="content-section visitor-section"
            aria-labelledby="visitors"
          >
            <SectionHeading id="visitors">Visitor Distribution</SectionHeading>
            <VisitorMap />
          </section>
        </div>
      </main>

      <footer className="site-footer">
        <p>© 2026 Yuang Wei</p>
        <p>Last updated: August 2026</p>
      </footer>
    </>
  );
}
