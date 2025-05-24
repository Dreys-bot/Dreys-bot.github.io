import { Section } from '@/partials/components/Section';
import { workExperiences } from '@/utils/WorkExperiences';

const WorkExperiences = () => (
    <Section>
      <div className="max-w-6xl mx-auto px-8">
        <h2 className="text-4xl font-bold text-white mb-12 text-center">WORK EXPERIENCE</h2>
        <div className="space-y-12">
          {workExperiences.map((exp) => (
            <div className="bg-gray-900/50 rounded-lg p-8 backdrop-blur-sm">
              <h3 className="text-2xl font-bold text-purple-400">{exp.company}</h3>
              <p className="text-xl text-gray-300 mt-2">Role: {exp.role}</p>
              <p className="text-lg text-gray-400 mt-4 italic">{exp.description}</p>
              <ul className="mt-6 space-y-3 text-gray-300 list-disc list-inside">
                {exp.details.map((detail) => (
                  <li>{detail}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </Section>
);
  
export { WorkExperiences };
  