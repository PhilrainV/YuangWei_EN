export type PublicationGroup = string;

export type Publication = {
  year: number;
  group: PublicationGroup;
  venue: string;
  title: string;
  authors: string;
  publication: string;
  webpage?: string;
  download: string;
  image?: string;
  imageFit?: "cover" | "contain";
};

export type OutputItem = {
  title: string;
  meta: string;
  url?: string;
  linkLabel?: string;
};

export type Honor = {
  year: string;
  title: string;
  award?: string;
};

export type Experience = {
  period: string;
  organization: string;
  role: string;
};

export type Profile = {
  name: string;
  chineseName: string;
  title: string;
  affiliation: string;
  affiliationUrl?: string;
  avatar: string;
  avatarAlt: string;
  location: string;
  email: string;
  bio: string[];
  researchInterests: string[];
  links: {
    researchGate: string;
    github: string;
    googleScholar: string;
    orcid: string;
  };
};

type AcademicContent = {
  profile?: Profile;
  publications?: Publication[];
  books?: OutputItem[];
  patents?: OutputItem[];
  softwareCopyrights?: OutputItem[];
  honors?: Honor[];
  experience?: Experience[];
};

declare global {
  interface Window {
    YUANG_WEI_CONTENT?: AcademicContent;
  }
}

const content = window.YUANG_WEI_CONTENT ?? {};

function list<T>(value: T[] | undefined): T[] {
  return Array.isArray(value) ? value : [];
}

export const publications = list(content.publications);
export const profile = content.profile as Profile;
export const books = list(content.books);
export const patents = list(content.patents);
export const softwareCopyrights = list(content.softwareCopyrights);
export const honors = list(content.honors);
export const experience = list(content.experience);
