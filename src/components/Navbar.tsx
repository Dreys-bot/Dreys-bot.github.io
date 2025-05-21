import React from 'react';
import { Link } from 'react-router-dom';

const Navbar: React.FC = () => {
  return (
    <nav className="fixed w-full top-0 z-50 flex justify-between items-center px-8 py-4 bg-transparent">
      <div className="text-white text-xl font-bold">
        <Link to="/">DJOUPE PENE Audrey</Link>
      </div>
      
      <div className="flex gap-8">
        <Link to="/about" className="text-white hover:text-purple-400 transition-colors">
          ABOUT
        </Link>
        <Link to="/projects" className="text-white hover:text-purple-400 transition-colors">
          Projects
        </Link>
        <Link to="/papers" className="text-white hover:text-purple-400 transition-colors">
          Papers
        </Link>
        <Link to="/certifications" className="text-white hover:text-purple-400 transition-colors">
          Certifications
        </Link>
        <a href="#contact" className="text-white hover:text-purple-400 transition-colors">
          CONTACT
        </a>
      </div>
    </nav>
  );
};

export default Navbar; 