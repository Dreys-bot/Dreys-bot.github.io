interface Experience {
    company: string;
    role: string;
    description: string;
    details: string[];
}

export const workExperiences: Experience[] = [
    {
      company: 'BpiFrance',
      role: 'Data Scientist Developer',
      description: 'Development of a Chatbot to Improve Financial Strategies',
      details: [
        'Data collection and structuring from various sources to prepare datasets.',
        'Selection and fine-tuning of Llama models to meet the client\'s specific needs.',
        'Backend Development: Integration of AI models into a secure and scalable architecture.',
        'Optimization of CPU and memory resources to ensure optimal performance.',
        'Security of sensitive data and secure integration into a Docker environment.',
        'Deployment of the system in production on an ultra-secure VPS.'
      ]
    },
    {
      company: 'GROUP VERDON',
      role: 'AI Engineer',
      description: 'Development of a personalized chatbot for the company',
      details: [
        'Analysis, processing and cleaning of the company\'s unstructured database for better use of information.',
        'Design and implementation of structured and optimized SQL databases for storing and managing data in vectors.',
        'Development of a custom intelligent chatbot, trained on internal company data using advanced LLM models (Llama and DeepSeek) locally.',
        'Chatbot optimization to ensure relevant responses using RAG and Search Engine methods.'
      ]
    },
    {
      company: 'TOTALEnergies',
      role: 'Data Scientist Consultant',
      description: 'Detection of emotion, gender and age on an embedded system to assess TotalEnergies customer satisfaction',
      details: [
        'Carrying out an in-depth benchmark of existing data and solutions to align the POC with TotalEnergies\' specific needs',
        'Selection and analysis of relevant images for training 3 detection models (age, gender and emotion)',
        'Training and optimizing each detection model on data for real-time operation',
        'Integration of the 3 detection models into the embedded system for extracting the age, emotion and gender of customers, then producing dynamic statistics on the results.'
      ]
    },
    {
      company: 'CAPGEMINI ENGINEERING',
      role: 'Data Scientist Consultant',
      description: 'Development of a model to assist in the diagnosis of patient illnesses',
      details: [
        'Carrying out a benchmark of existing data and solutions to align the methods to be used with the needs of the project.',
        'Collection and implementation of an ETL pipeline for processing more than 10,000 radiographic images improving their quality using Azure Data factory, Azure Synapse Analytics, Azure Data Lake Gen 2, Databricks, Azure Key Vault.',
        'Data visualization and rebalancing for analysis to optimize model training.',
        'Development of a high-performance AI model (87% accuracy) integrated into a web application for doctors, with a CI/CD pipeline for fast and reliable deployments (Docker, Git, AWS), ensuring fast and continuous updates.'
      ]
    }
];