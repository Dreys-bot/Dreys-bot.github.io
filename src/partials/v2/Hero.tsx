import { Section } from '@/partials/components/Section';

const Hero = () => (
    <Section>
        <div className="max-w-6xl mx-auto px-8">
            <div className="grid md:grid-cols-2 gap-12 items-start">
                <div className="relative">
                <img
                    src="/assets/images/me/tof2.jpeg"
                    alt="DJOUPE PENE Audrey"
                    className="rounded-lg shadow-xl sticky top-24"
                    width={500}
                    height={600}
                    style={{objectFit: "cover"}}
                />
                </div>
                
                <div className="space-y-6 text-white">
                <h1 className="text-4xl font-bold mb-4">About Me</h1>
                <p className="text-lg text-gray-300">
                    Hello! I'm Audrey DJOUPE PENE, a passionate Data Scientist and Machine Learning Engineer. 
                    I specialize in developing innovative solutions using cutting-edge AI technologies.
                </p>
                
                <div className="space-y-4">
                    <h2 className="text-2xl font-semibold text-white">Skills</h2>
                    <ul className="list-disc list-inside text-gray-300 space-y-2">
                    <li>Machine Learning & Deep Learning</li>
                    <li>Data Analysis & Visualization</li>
                    <li>Python, TensorFlow, PyTorch</li>
                    <li>Natural Language Processing</li>
                    <li>Statistical Analysis</li>
                    </ul>
                </div>
                </div>
            </div>
        </div>
    </Section>
);
  
  export { Hero };
  