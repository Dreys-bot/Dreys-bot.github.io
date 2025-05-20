import type { MarkdownHeading } from 'astro';
import type {
  IFrontmatter,
  MarkdownInstance,
} from 'astro-boilerplate-components';
import { useEffect, useRef, useState } from 'react';

export const sortByDate = (posts: MarkdownInstance<IFrontmatter>[]) => {
  return posts.sort(
    (a, b) =>
      new Date(b.frontmatter.pubDate).valueOf() -
      new Date(a.frontmatter.pubDate).valueOf()
  );
};

export type THeading = {
  depth: number;
  text: string;
  slug: string;
  subheadings: THeading[];
};

export const generateToc = (headings: MarkdownHeading[]): THeading[] => {
  const toc: THeading[] = [];
  const parentHeadings = new Map();

  headings.forEach((h) => {
    const heading = { ...h, subheadings: [] };
    parentHeadings.set(heading.depth, heading);

    if (heading.depth === 1) {
      toc.push(heading);
    } else {
      parentHeadings.get(heading.depth - 1).subheadings.push(heading);
    }
  });
  return toc;
};

export function useHeadsObserver() {
  const observer = useRef(null);
  const [activeId, setActiveId] = useState('');

  useEffect(() => {
    const handleObsever = (entries) => {
      entries.forEach((entry) => {
        if (entry?.isIntersecting) {
          setActiveId(entry.target.id);
        }
      });
    };

    observer.current = new IntersectionObserver(handleObsever, {
      rootMargin: '-20px 0px -20px 0px',
    });

    const elements = document.querySelectorAll('.prose h1, .prose h2');
    elements.forEach((elem) => observer.current.observe(elem));
    return () => observer.current?.disconnect();
  }, []);

  return { activeId };
}
