import json

data = [
    {"id": "vec1", "text": "Microsoft is known for its software products, including the Windows operating system and Office suite."},
    {"id": "vec2", "text": "Google, the tech giant, dominates the search engine market and has products like Gmail, Android, and YouTube."},
    {"id": "vec3", "text": "Tesla is a company focused on electric vehicles, renewable energy solutions, and autonomous driving technology."},
    {"id": "vec4", "text": "Amazon started as an online bookstore and has since become a leading e-commerce and cloud computing company."},
    {"id": "vec5", "text": "Facebook, now Meta, is a social media platform that connects billions of users worldwide."},
    {"id": "vec6", "text": "Apple Inc. is a multinational technology company, famous for its iPhones, Mac computers, and services like iCloud."},
    {"id": "vec7", "text": "IBM is known for its innovations in computing hardware, software, and IT services, particularly in artificial intelligence."},
    {"id": "vec8", "text": "SpaceX, founded by Elon Musk, is a private aerospace manufacturer and space transport services company."},
    {"id": "vec9", "text": "Netflix is a streaming service that offers movies, TV shows, and original content for subscribers worldwide."},
    {"id": "vec10", "text": "Nike is a global leader in athletic apparel, footwear, and sports equipment, well-known for its 'Just Do It' slogan."},
    {"id": "vec11", "text": "Toyota is a Japanese automobile manufacturer known for producing reliable and fuel-efficient cars."},
    {"id": "vec12", "text": "Zoom Video Communications provides cloud-based video conferencing services used by businesses and individuals globally."},
    {"id": "vec13", "text": "Adobe Systems is known for its creative software products, including Photoshop, Illustrator, and Acrobat."},
    {"id": "vec14", "text": "Spotify is a music streaming service offering millions of songs, playlists, and podcasts for its users."},
    {"id": "vec15", "text": "Intel is a leading semiconductor company that designs and manufactures processors for computers and other electronic devices."},
    {"id": "vec16", "text": "Samsung is a South Korean multinational conglomerate known for its electronics, especially smartphones, televisions, and home appliances."},
    {"id": "vec17", "text": "Uber is a transportation network company that allows users to book rides through its mobile app."},
    {"id": "vec18", "text": "Twitter, now X, is a social media platform that enables users to post short messages called tweets."},
    {"id": "vec19", "text": "Alibaba Group is a Chinese multinational specializing in e-commerce, retail, internet, and technology services."},
    {"id": "vec20", "text": "Coca-Cola is a multinational beverage corporation known for its iconic soda drinks like Coca-Cola, Diet Coke, and Fanta."},
    {"id": "vec21", "text": "Spotify offers personalized music recommendations and playlists using machine learning algorithms to tailor to users' tastes."},
    {"id": "vec22", "text": "Boeing is an aerospace company that designs and manufactures commercial and military aircraft, satellites, and defense systems."},
    {"id": "vec23", "text": "LinkedIn is a professional networking platform used for job searches, networking, and sharing business-related content."},
    {"id": "vec24", "text": "TikTok is a social media platform known for short, viral videos often set to music, and its algorithm-driven content discovery."},
    {"id": "vec25", "text": "Spotify's algorithm personalizes music for listeners based on their previous listening habits and preferences."},
    {"id": "vec26", "text": "Alibaba's e-commerce platform includes sites like Taobao and Tmall, which cater to Chinese consumers and global sellers."},
    {"id": "vec27", "text": "Salesforce is a cloud-based software company providing customer relationship management (CRM) services and applications."},
    {"id": "vec28", "text": "Snapchat is a multimedia messaging app known for its disappearing messages and augmented reality features."},
    {"id": "vec29", "text": "The iPhone is a line of smartphones designed and marketed by Apple, known for its sleek design and iOS ecosystem."},
    {"id": "vec30", "text": "Zoom's video conferencing service became especially popular during the COVID-19 pandemic for remote work and education."},
    {"id": "vec31", "text": "Oracle provides software solutions including database management systems, enterprise resource planning (ERP), and cloud computing services."},
    {"id": "vec32", "text": "Slack is a collaboration platform used by businesses to communicate via messaging, file sharing, and project management tools."},
    {"id": "vec33", "text": "WhatsApp is a messaging app that allows users to send text, voice, and video messages over the internet."},
    {"id": "vec34", "text": "Pinterest is a visual discovery platform where users can find and save ideas for various projects and interests."},
    {"id": "vec35", "text": "Airbnb is an online marketplace for lodging and travel experiences, allowing people to book homes and unique stays."},
    {"id": "vec36", "text": "TikTok's algorithm uses machine learning to recommend short-form video content based on user interaction and engagement."},
    {"id": "vec37", "text": "Walmart is a multinational retail corporation that operates large chain stores offering a wide variety of products."},
    {"id": "vec38", "text": "Tesla's electric cars, like the Model S, Model 3, and Model X, are designed to reduce environmental impact and promote clean energy."},
    {"id": "vec39", "text": "NVIDIA is a technology company known for its graphics processing units (GPUs) used in gaming, AI, and high-performance computing."},
    {"id": "vec40", "text": "TikTok's viral trends and challenges often involve dances, comedy, and lip-syncing to popular music tracks."}
]

# Save the data to a JSON file
with open('company_data.json', 'w') as file:
    json.dump(data, file, indent=4) 