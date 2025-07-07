# 🛒 E-Commerce Data Acquisition, Analysis, and Network Visualization Tool

DataWizards - DSAI 103 Project:
A comprehensive tool for scraping, analyzing, and visualizing e-commerce data from eBay using multiple acquisition methods and advanced network analysis techniques.

🚀 Features
Data Acquisition:
- Multi-method scraping: BeautifulSoup, Selenium, and eBay API
- Category-based search: Electronics, Fashion, Collectibles, and more
- Custom keyword search: Target specific products
- Relationship extraction: "Frequently Bought Together" analysis

Network Analysis:
- Graph construction: Product relationship networks
- Centrality analysis: Degree and betweenness centrality
- Community detection: Identify product clusters
- Interactive visualization: NetworkX-powered graphs

Visualizations:
- 2D Analytics: Price distribution, category analysis
- Interactive heatmaps: Price vs. category/region
- 3D point clouds: Multi-dimensional product mapping
- Real-time filtering: Dynamic data exploration

User Interface:
- Streamlit dashboard: Clean, intuitive interface
- Multi-tab layout: Organized workflow
- Progress tracking: Real-time scraping progress
- Data export: CSV file generation

📋 Prerequisites
System Requirements:
- Python 3.7+
- Chrome browser (for Selenium)
- Internet connection
eBay API Access:
- eBay Developer Account
- Client ID and Client Secret
- Replace credentials in the configuration section

🔧 Installation
1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/ecommerce-data-analysis.git
   cd ecommerce-data-analysis
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Configure eBay API (Optional)
   - Edit the configuration section in the main script
   - Replace `CLIENT_ID` and `CLIENT_SECRET` with your credentials

📦 Dependencies
```
requests
beautifulsoup4
selenium
streamlit
pandas
numpy
networkx
matplotlib
seaborn
webdriver-manager
```

🏃‍♂️ Usage
Running the Application:
1. Start the Streamlit app
   ```bash
   streamlit run main.py
   ```
2. Access the dashboard
   - Open your browser to `http://localhost:8501`
Using the Interface:
1. Data Acquisition Tab
- Select acquisition methods: Choose from BeautifulSoup, Selenium, or eBay API
- Choose categories: Electronics, Fashion, Collectibles, etc.
- Set item limits: Control the number of products per category
- Custom keywords: Search for specific products
- Start scraping: Begin data collection process
2. Network Analysis Tab
- View network graphs: Product relationship visualization
- Analyze centrality: Identify influential products
- Community detection: Discover product clusters
- Interactive metrics: Real-time network statistics
3. Visualizations Tab
- Price analysis: Category-wise price distribution
- Interactive heatmaps: Multi-dimensional analysis
- Data filtering: Dynamic table exploration
- Export options: Download filtered data
4. 3D Point Cloud Tab
- 3D visualization: Price vs. Reviews vs. Rating
- Interactive plots: Rotate and zoom
- Pattern recognition: Identify product clusters

🗂️ Data Structure
Product Data:
```json
{
  "itemId": "unique_identifier",
  "title": "Product Title",
  "price": "$99.99",
  "priceValue": 99.99,
  "category": "Electronics",
  "link": "product_url",
  "image": "image_url",
  "rating": "4.5/5",
  "seller": "seller_name",
  "location": "USA",
  "source": "BeautifulSoup|Selenium|API"
}
```
Relationship Data:
```json
{
  "Product1": "product_id_1",
  "Product2": "product_title_2",
  "Frequency": 1,
  "Relation": "Frequently Bought Together"
}
```

📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact
DataWizards Team
- Author: Almoatasembellah, Amir
- Email: [s-almoatasembellah.gafer@zewailcity.edu.eg]
         [s-amir.elsamahy@zewailcity.edu.eg]
