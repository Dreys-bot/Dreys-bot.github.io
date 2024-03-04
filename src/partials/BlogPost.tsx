import type { MarkdownHeading } from 'astro';
import type { IFrontmatter } from 'astro-boilerplate-components';
import { PostContent, PostHeader, Section } from 'astro-boilerplate-components';
import type { ReactNode } from 'react';

import { AppConfig } from '@/utils/AppConfig';

import TableOfContent from './TableOfContent';

type IBlogPostProps = {
  frontmatter: IFrontmatter;
  headings: MarkdownHeading[];
  children: ReactNode;
};

const BlogPost = (props: IBlogPostProps) => (
  <Section>
    <PostHeader content={props.frontmatter} author={AppConfig.author} />

    <div className="mt-8 grid grid-cols-1 gap-10 md:grid-cols-[20%_auto]">
      {/* Aside */}
      <TableOfContent headings={props.headings} />

      {/* Article */}
      <PostContent content={props.frontmatter}>{props.children}</PostContent>
    </div>
  </Section>
);

export { BlogPost };
