type INavMenuItemProps = {
    href: string;
    children: string;
    target?: '_blank' | '_self' | '_parent' | '_top' | string;
};

const NavMenuItem = (props: INavMenuItemProps) => (
    <li className="text-white hover:text-purple-400 transition-colors">
        <a href={props.href} target={props.target || '_self'}>{props.children}</a>
    </li>
);

export { NavMenuItem };