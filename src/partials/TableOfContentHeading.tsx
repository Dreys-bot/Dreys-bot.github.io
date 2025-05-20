import type { THeading } from '@/utils/Posts';
import { useHeadsObserver } from '@/utils/Posts';

interface ITOCHProps {
  heading: THeading;
}

const TableOfContentHeading = (props: ITOCHProps) => {
  const { heading } = props;
  const { activeId } = useHeadsObserver();

  const TOCLink = ({ link, label }) => (
    <a
      href={`#${link}`}
      className="mb-2 line-clamp-2 w-fit rounded-md bg-slate-200 px-4 py-1 first-letter:uppercase hover:bg-indigo-300 hover:text-white dark:bg-slate-800 dark:text-white dark:hover:bg-indigo-400"
      style={{
        backgroundColor:
          activeId === link ? 'rgb(129, 140, 248)' : 'rgb(30, 41, 59)',
      }}
      onClick={(e) => {
        e.preventDefault();

        document.querySelector(`#${link}`).scrollIntoView({
          behavior: 'smooth',
        });
      }}
    >
      {label}
    </a>
  );

  return (
    <li className="flex flex-col">
      <TOCLink link={heading.slug} label={heading.text} />

      {heading.subheadings.length > 0 && (
        <ul className="ml-8">
          {heading.subheadings.map((subheading) => (
            <TOCLink link={subheading.slug} label={subheading.text} />
          ))}
        </ul>
      )}
    </li>
  );
};

export default TableOfContentHeading;
