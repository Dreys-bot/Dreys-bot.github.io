import React from 'react';
import backgroundImage from '../assets/tof1.jpeg';

const Hero: React.FC = () => {
  return (
    <div className="relative h-screen w-full">
      {/* Background image */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ 
          backgroundImage: `url(${backgroundImage})`,
          filter: 'brightness(0.7)'
        }}
      />
      
      {/* Content overlay */}
      <div className="relative z-10 h-full flex items-center justify-start px-8">
        <div className="bg-black bg-opacity-40 p-8 rounded-lg max-w-2xl backdrop-blur-sm">
          <h1 className="text-4xl md:text-6xl text-white mb-4 font-bold">
            👋 👋 HEY, I'M DJOUPE Audrey
          </h1>
          <div className="text-xl md:text-2xl text-white space-y-2">
            <p>Data Science → ML & DL Engineer</p>
            <p>Writer</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero; 