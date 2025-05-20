import type { MarkdownHeading } from 'astro';

import type { THeading } from '@/utils/Posts';
import { generateToc } from '@/utils/Posts';

import TableOfContentHeading from './TableOfContentHeading';

interface ITOCProps {
  headings: MarkdownHeading[];
}

const TableOfContent = (props: ITOCProps) => {
  const toc: THeading[] = generateToc(props.headings);

  return (
    <nav className="toc sticky top-24 hidden max-w-xs self-start transition-all duration-200 dark:text-black md:block">
      <h1 className="mb-3 text-2xl font-bold dark:text-white">Index</h1>
      <ul className="flex flex-col gap-1 [text-wrap:balance]">
        {toc.map((heading) => (
          <TableOfContentHeading heading={heading} />
        ))}
      </ul>
    </nav>
  );
};

export default TableOfContent;
