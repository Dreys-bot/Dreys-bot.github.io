import React from 'react';

interface Experience {
  company: string;
  role: string;
  description: string;
}

const experiences: Experience[] = [
  {
    company: 'BpiFrance',
    role: 'Data Scientist Developer',
    description: 'Development of a Chatbot to Improve Financial Strategies'
  },
  // You can add more experiences here
];

const WorkExperience: React.FC = () => {
  return (
    <section className="py-20 px-8 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold text-center mb-16 relative">
          WORK EXPERIENCE
          <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-20 h-1 bg-purple-500 mt-4"></div>
        </h2>
        
        <div className="space-y-12">
          {experiences.map((exp, index) => (
            <div key={index} className="bg-white p-8 rounded-lg shadow-lg">
              <h3 className="text-2xl font-bold text-purple-600">{exp.company}</h3>
              <p className="text-lg font-medium mt-2">Role: {exp.role}</p>
              <p className="mt-4 text-gray-600">{exp.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default WorkExperience; 