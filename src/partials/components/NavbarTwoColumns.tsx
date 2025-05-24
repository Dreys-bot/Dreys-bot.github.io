import type { ReactNode } from 'react';

type INavbarProps = {
  children: ReactNode;
};

const NavbarTwoColumns = (props: INavbarProps) => (
  <nav className="fixed w-full top-0 z-50 flex justify-between items-center px-8 py-4 bg-gray-900/80 backdrop-blur-sm">
    {props.children}
  </nav>
);

export { NavbarTwoColumns };