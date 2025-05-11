"""
E-Commerce Data Acquisition, Analysis, and Network Visualization Tool
Author: [Almoatasembellah, Amir]

Features:
- Multi-method data acquisition (BeautifulSoup, Selenium, API)
- Advanced network analysis with NetworkX
- 2D visualization (heatmaps, charts)
- 3D point cloud visualization
- Interactive UI with Streamlit
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import json
import random
import base64
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ==============================================
# CONFIGURATION SETTINGS
# ==============================================

# eBay API credentials - replace with your own
CLIENT_ID = "Moatasem-da-PRD-10ae609bd-03482525"
CLIENT_SECRET = "PRD-0ae609bd718f-2f22-44f3-a346-31ec"
MARKETPLACE_ID = "EBAY_US"

# Categories to scrape
CATEGORIES = [
    "Electronics", "Computers", "Smart Home", 
    "Fashion", "Collectibles", "Sports",
    "Health & Beauty", "Home & Garden"
]

# User agent to avoid getting blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/"
}

# File paths
PRODUCTS_CSV = "ebay_products.csv"
RELATIONSHIPS_CSV = "ebay_relationships.csv"

# ==============================================
# DATA ACQUISITION METHODS
# ==============================================

def get_ebay_access_token():
    """Get eBay OAuth access token"""
    try:
        encoded_auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode("utf-8")).decode("utf-8")
        url = "https://api.ebay.com/identity/v1/oauth2/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {encoded_auth}"
        }
        data = {
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope"
        }
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        token_info = response.json()
        return token_info["access_token"]
    except Exception as e:
        st.error(f"Failed to get eBay access token: {e}")
        return None

def soup_scraper(category, pages=1):
    """Scrape eBay using BeautifulSoup"""
    ebay_url = "https://www.ebay.com/sch/i.html?_nkw="
    products = []

    for page in range(1, pages + 1):
        url = f"{ebay_url}{category.replace(' ', '+')}&_pgn={page}"
        
        try:
            response = requests.get(url, headers=HEADERS)
            
            if response.status_code != 200:
                st.warning(f"Failed to fetch {url}, status code: {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            for item in soup.select('.s-item'):
                title = item.select_one('.s-item__title')
                price = item.select_one('.s-item__price')
                link = item.select_one('.s-item__link')
                image = item.select_one('.s-item__image-img')
                rating = item.select_one('.s-item__reviews-count')
                seller = item.select_one('.s-item__seller-info-text')
                location = item.select_one('.s-item__location')
                
                if title and price and link:
                    # Clean the price text and convert to numeric
                    price_text = price.text.strip()
                    price_value = None
                    
                    if price_text and price_text != "":
                        try:
                            # Remove currency symbol and commas, then convert to float
                            price_value = float(price_text.replace('$', '').replace(',', '').split(' to ')[0])
                        except (ValueError, TypeError):
                            price_value = None

                    products.append({
                        "itemId": link["href"].split('itm/')[1].split('?')[0] if 'itm/' in link["href"] else f"item-{len(products)}",
                        "title": title.text.strip(),
                        "price": price_text,
                        "priceValue": price_value,
                        "category": category,
                        "link": link["href"],
                        "image": image["src"] if image else None,
                        "rating": rating.text.strip() if rating else "No rating",
                        "seller": seller.text.strip() if seller else "Unknown",
                        "location": location.text.strip() if location else "Unknown",
                        "source": "BeautifulSoup"
                    })

            # Avoid getting blocked
            time.sleep(2)
        
        except Exception as e:
            st.error(f"Error in BeautifulSoup scraping for {category}: {e}")
    
    return products

def selenium_scraper(category, max_items=10):
    """Scrape eBay using Selenium for dynamic content"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    
    products = []
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        url = f"https://www.ebay.com/sch/i.html?_nkw={category.replace(' ', '+')}"  
        driver.get(url)
        
        # Wait for JavaScript elements to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "s-item")))
        
        # Scroll to load more items
        for i in range(3):  # Scroll 3 times
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(2)
        
        items = driver.find_elements(By.CLASS_NAME, "s-item")
        
        # Limit the number of items to process
        for item in items[:max_items]:
            try:
                title_element = item.find_elements(By.CLASS_NAME, "s-item__title")
                price_element = item.find_elements(By.CLASS_NAME, "s-item__price")
                link_element = item.find_elements(By.CLASS_NAME, "s-item__link")
                image_element = item.find_elements(By.CLASS_NAME, "s-item__image-img")
                rating_element = item.find_elements(By.CLASS_NAME, "s-item__reviews-count")
                
                title = title_element[0].text.strip() if title_element else "No title"
                price_text = price_element[0].text.strip() if price_element else "No price"
                link = link_element[0].get_attribute("href") if link_element else "No link"
                image = image_element[0].get_attribute("src") if image_element else "No image"
                rating = rating_element[0].text.strip() if rating_element else "No rating"
                
                # Parse the price value
                price_value = None
                if price_text and price_text != "No price":
                    try:
                        price_value = float(price_text.replace('$', '').replace(',', '').split(' to ')[0])
                    except (ValueError, TypeError):
                        price_value = None
                
                item_id = link.split('itm/')[1].split('?')[0] if link != "No link" and 'itm/' in link else f"sel-{len(products)}"
                
                products.append({
                    "itemId": item_id,
                    "title": title,
                    "price": price_text,
                    "priceValue": price_value,
                    "category": category,
                    "link": link,
                    "image": image,
                    "rating": rating,
                    "seller": "Unknown", # Could extract with more Selenium code
                    "location": "Unknown", # Could extract with more Selenium code
                    "source": "Selenium"
                })
            
            except Exception as e:
                st.warning(f"Error extracting item data with Selenium: {e}")
                continue
        
        driver.quit()
    
    except Exception as e:
        st.error(f"Error in Selenium scraping for {category}: {e}")
    
    return products

def api_scraper(category, access_token, max_items=10):
    """Scrape eBay using their official API"""
    if not access_token:
        return []
        
    url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    params = {
        "q": category,
        "limit": max_items
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE_ID,
        "Content-Type": "application/json"
    }
    
    products = []
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            st.warning(f"API error: {response.status_code} - {response.text}")
            return []
        
        data = response.json()
        items = data.get("itemSummaries", [])
        
        for item in items:
            # Extract needed fields from API response
            item_id = item.get("itemId", f"api-{len(products)}")
            title = item.get("title", "No title")
            link = item.get("itemWebUrl", "No link")
            image = item.get("image", {}).get("imageUrl", "No image")
            
            # Price handling
            price_info = item.get("price", {})
            price_currency = price_info.get("currency", "USD")
            price_value_str = price_info.get("value", "0")
            
            try:
                price_value = float(price_value_str)
                price_text = f"{price_currency} {price_value}"
            except (ValueError, TypeError):
                price_value = None
                price_text = "No price"
            
            # Location and shipping
            seller_info = item.get("seller", {})
            seller_name = seller_info.get("username", "Unknown")
            
            location_info = item.get("itemLocation", {})
            location = location_info.get("country", "Unknown")
            
            # Rating - not always provided by API
            rating = "No rating"
            if "seller" in item and "feedbackPercentage" in item["seller"]:
                rating = f"{item['seller']['feedbackPercentage']}% positive"
            
            products.append({
                "itemId": item_id,
                "title": title,
                "price": price_text,
                "priceValue": price_value,
                "category": category,
                "link": link,
                "image": image,
                "rating": rating,
                "seller": seller_name,
                "location": location,
                "source": "API"
            })
    
    except Exception as e:
        st.error(f"Error in API scraping for {category}: {e}")
    
    return products

def scrape_subcategories(main_category):
    """Scrape subcategories from eBay"""
    url = f"https://www.ebay.com/b/{main_category.replace(' ', '-')}"
    subcategories = {}
    
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find subcategory links - adjust selectors based on eBay's structure
        subcat_elements = soup.find_all('a', class_='textual-display brw-category-nav__link')
        
        for element in subcat_elements:
            name = element.text.strip()
            link = element['href']
            subcategories[name] = link
            
    except Exception as e:
        st.warning(f"Error scraping subcategories for {main_category}: {e}")
    
    return subcategories

def scrape_frequently_bought_together(products_data):
    """Scrape 'Frequently Bought Together' recommendations using Selenium"""
    if not products_data:
        return []
    
    relationships = []
    
    try:
        options = Options()
        # options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Process a subset of products to avoid long run times
        sample_size = min(10, len(products_data))
        for product in products_data[:sample_size]:
            product_id = product["itemId"]
            product_url = product["link"]
            
            if product_url == "No link":
                continue
                
            try:
                driver.get(product_url)
                time.sleep(2)  # Wait for page to load
                
                # Find frequently bought together section - adjust selectors based on eBay's structure
                carousel_items = driver.find_elements(By.CSS_SELECTOR, "ul.carousel__list li.carousel__snap-point")
                
                if not carousel_items:
                    # Try alternative selectors if the first one doesn't work
                    carousel_items = driver.find_elements(By.CSS_SELECTOR, ".ux-similar-products")
                
                for item in carousel_items:
                    try:
                        title_element = item.find_element(By.CSS_SELECTOR, "h3")
                        related_title = title_element.text.strip()
                        
                        if related_title and related_title != product["title"]:
                            relationships.append({
                                "Product1": product_id,
                                "Product2": related_title,
                                "Frequency": 1
                            })
                    except:
                        continue
            
            except Exception as e:
                st.warning(f"Error scraping FBT for {product_url}: {e}")
                continue
        
        driver.quit()
    
    except Exception as e:
        st.error(f"Error in FBT scraping: {e}")
    
    return relationships

# ==============================================
# DATA PROCESSING AND STORAGE
# ==============================================

def save_to_csv(data, filename):
    """Save data to CSV file"""
    if not data:
        st.warning(f"No data to save to {filename}")
        return False
    
    try:
        # Get fieldnames from the first item, ensuring all possible fields are included
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            
        st.success(f"Data saved to {filename}")
        return True
    
    except Exception as e:
        st.error(f"Error saving data to {filename}: {e}")
        return False

def load_from_csv(filename):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        st.error(f"Error loading data from {filename}: {e}")
        return pd.DataFrame()

def preprocess_product_data(products):
    """Clean and preprocess product data"""
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(products)
    
    # Remove duplicates based on itemId
    df = df.drop_duplicates(subset=['itemId'])
    
    # Handle missing values
    df['title'] = df['title'].fillna('Unknown Title')
    df['price'] = df['price'].fillna('No price')
    df['category'] = df['category'].fillna('Uncategorized')
    
    # Create numeric price column if not exists
    if 'priceValue' not in df.columns:
        df['priceValue'] = pd.to_numeric(df['price'].str.replace('$', '').str.replace(',', ''), errors='coerce')
    
    # Add region if not exists
    if 'region' not in df.columns:
        df['region'] = df['location'].fillna('Unknown')
    
    # Add condition if not exists (simulate with random values for demo)
    if 'condition' not in df.columns:
        conditions = ['New', 'Used - Like New', 'Used - Good', 'Used - Acceptable', 'Refurbished']
        df['condition'] = [random.choice(conditions) for _ in range(len(df))]
    
    # Add reviews count if not exists (simulate with random values for demo)
    if 'reviews' not in df.columns:
        df['reviews'] = np.random.randint(1, 500, size=len(df))
    
    # Convert back to list of dicts
    return df.to_dict('records')

def create_relationships_data(products, relationships):
    """Create or enhance relationships data"""
    if not relationships or len(relationships) < 3:  # Check if relationships data is insufficient
        # If no relationships data, create some based on category and price similarity
        df = pd.DataFrame(products)
        relationships = []
        
        # Add proper error handling for DataFrame operations
        try:
            # Ensure priceValue is numeric
            df['priceValue'] = pd.to_numeric(df['priceValue'], errors='coerce')
            df = df.dropna(subset=['priceValue']).reset_index(drop=True)
            
            # Group by category
            for category, group in df.groupby('category'):
                if len(group) < 2:
                    continue
                    
                # Sort by price to find similar items
                group = group.sort_values('priceValue').reset_index(drop=True)
                
                # Create relationships between consecutive items by price
                for i in range(len(group) - 1):
                    if i + 1 < len(group):
                        relationships.append({
                            'Product1': group.iloc[i]['itemId'],
                            'Product2': group.iloc[i+1]['title'],
                            'Frequency': 1,
                            'Relation': 'Similar Price'
                        })
                
                # Create some relationships between random items in the same category
                for _ in range(min(5, len(group))):
                    if len(group) >= 2:  # Need at least 2 items to create a relationship
                        idx1, idx2 = random.sample(range(len(group)), 2)
                        relationships.append({
                            'Product1': group.iloc[idx1]['itemId'],
                            'Product2': group.iloc[idx2]['title'],
                            'Frequency': 1,
                            'Relation': 'Same Category'
                        })
        except Exception as e:
            st.warning(f"Error creating relationships data: {e}")
            
            # If all else fails, create some minimal dummy relationships
            if len(products) >= 2:
                for i in range(min(10, len(products)-1)):
                    relationships.append({
                        'Product1': products[i].get('itemId', f'item-{i}'),
                        'Product2': products[i+1].get('title', f'Product {i+1}'),
                        'Frequency': 1,
                        'Relation': 'Fallback Relation'
                    })
    
    return relationships

# ==============================================
# NETWORK ANALYSIS AND VISUALIZATION
# ==============================================


def perform_network_analysis(relationships_data):
    """Perform network analysis using NetworkX, focusing on degree and betweenness only"""
    # Convert relationships to DataFrame if it's a list
    if isinstance(relationships_data, list):
        if not relationships_data:
            st.warning("No relationship data for network analysis")
            return None, None, None
        relationships_df = pd.DataFrame(relationships_data)
    else:
        relationships_df = relationships_data
    
    # Create graph
    G = nx.Graph()
    
    # Add edges from relationships
    for _, row in relationships_df.iterrows():
        prod1 = str(row["Product1"])
        prod2 = str(row["Product2"])
        
        # Add weight if available
        if 'Frequency' in relationships_df.columns:
            try:
                weight = float(row["Frequency"])
            except (ValueError, TypeError):
                weight = 1.0
        else:
            weight = 1.0
            
        G.add_edge(prod1, prod2, weight=weight)
    
    # Calculate basic network metrics
    metrics = {
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": nx.density(G),
        "Average Clustering": nx.average_clustering(G)
    }
    
    # Calculate centrality measures with error handling
    centrality = {}
    try:
        centrality["Degree"] = nx.degree_centrality(G)
    except Exception as e:
        st.warning(f"Could not calculate degree centrality: {e}")
        centrality["Degree"] = {node: 0.1 for node in G.nodes()}
        
    try:
        centrality["Betweenness"] = nx.betweenness_centrality(G)
    except Exception as e:
        st.warning(f"Could not calculate betweenness centrality: {e}")
        centrality["Betweenness"] = {node: 0.1 for node in G.nodes()}
    
    return G, metrics, centrality

def visualize_network(G, centrality=None, title="Product Network"):
    """Visualize network using NetworkX and Matplotlib"""
    if not G:
        st.warning("No graph to visualize")
        return None
    
    # Handle empty graph
    if G.number_of_nodes() == 0:
        st.warning("Graph has no nodes to visualize")
        return None
    
    plt.figure(figsize=(12, 10))
    
    # Use spring layout with increased iterations for better stability
    try:
        pos = nx.spring_layout(G, seed=42, iterations=100)
    except Exception:
        # Fall back to simpler layout algorithm if spring layout fails
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            # Last resort: use random layout
            pos = nx.random_layout(G)
    
    # Set node sizes based on centrality if provided
    if centrality and isinstance(centrality, dict):
        # Ensure all nodes have a centrality value
        default_size = 0.1
        node_sizes = [3000 * centrality.get(node, default_size) for node in G.nodes()]
    else:
        node_sizes = 300
    
    # Try to detect communities
    try:
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(G))
        
        # Color nodes by community
        colors = []
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
                
        colors = [community_map.get(node, 0) for node in G.nodes()]
        
        # Use a try block for drawing to handle any matplotlib issues
        try:
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, cmap=plt.cm.tab20, alpha=0.8)
        except Exception:
            # Fall back to simple node drawing
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    except Exception:
        # Fall back to simple visualization if community detection fails
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    
    # Draw edges with varying width based on weight
    try:
        edge_weights = [G[u][v].get('weight', 1.0) * 1.5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    except Exception:
        # Fall back to simple edge drawing
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
    
    # Add labels for most central nodes only (to avoid clutter)
    if centrality and isinstance(centrality, dict):
        try:
            # Sort nodes by centrality and label only top 10
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            labels = {node: node for node, _ in top_nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
        except Exception:
            # Fall back to simpler labeling
            if G.number_of_nodes() <= 10:
                nx.draw_networkx_labels(G, pos, font_size=8)
    else:
        # Label all nodes if there are not too many
        if G.number_of_nodes() <= 20:
            try:
                nx.draw_networkx_labels(G, pos, font_size=8)
            except Exception:
                pass
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    return plt

def create_pricing_heatmap(products_data, groupby_columns=['category', 'region']):
    """Create heatmap showing price distribution by categories and regions"""
    df = pd.DataFrame(products_data)
    
    # Ensure price is numeric
    df['priceValue'] = pd.to_numeric(df['priceValue'], errors='coerce')
    
    # Drop records with missing prices
    df = df.dropna(subset=['priceValue'])
    
    if df.empty:
        st.warning("No valid pricing data available for heatmap")
        return None
    
    # Create pivot table based on provided groupby columns
    pivot_table = pd.pivot_table(
        df, 
        values="priceValue", 
        index=groupby_columns[0], 
        columns=groupby_columns[1], 
        aggfunc="mean"
    )
    
    if pivot_table.empty:
        st.warning("Pivot table is empty; unable to generate heatmap")
        return None
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Average Price by {groupby_columns[0].title()} and {groupby_columns[1].title()}")
    plt.ylabel(groupby_columns[0].title())
    plt.xlabel(groupby_columns[1].title())
    plt.tight_layout()
    
    return plt

def create_point_cloud(products_data):
    """Create 3D point cloud visualization"""
    df = pd.DataFrame(products_data)
    
    # Ensure required columns exist
    df['priceValue'] = pd.to_numeric(df['priceValue'], errors='coerce')
    
    # Generate or ensure other numeric dimensions exist
    if 'reviews' not in df.columns:
        df['reviews'] = np.random.randint(1, 500, size=len(df))
    else:
        df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce')
    
    if 'rating_numeric' not in df.columns:
        # Extract numeric rating from rating text or generate random
        df['rating_numeric'] = np.random.uniform(1.0, 5.0, size=len(df))
        df['rating_numeric'] = np.round(df['rating_numeric'], 1)
    
    # Drop records with missing values
    df = df.dropna(subset=['priceValue', 'reviews', 'rating_numeric'])
    
    if len(df) < 5:
        st.warning("Not enough data points for 3D visualization")
        return None
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize data for better visualization
    price_norm = (df['priceValue'] - df['priceValue'].min()) / (df['priceValue'].max() - df['priceValue'].min() + 0.1)
    reviews_norm = (df['reviews'] - df['reviews'].min()) / (df['reviews'].max() - df['reviews'].min() + 0.1)
    
    # Create scatter plot
    scatter = ax.scatter(
        price_norm,
        reviews_norm,
        df['rating_numeric'],
        c=df['rating_numeric'],
        cmap='viridis',
        s=50 * (price_norm + 0.5),  # Size based on price
        alpha=0.7
    )
    
    # Add labels and colorbar
    ax.set_xlabel('Price (normalized)')
    ax.set_ylabel('Reviews Count (normalized)')
    ax.set_zlabel('Rating (1-5)')
    ax.set_title('3D Point Cloud: Price vs Reviews vs Rating')
    fig.colorbar(scatter, ax=ax, label='Rating')
    
    # Add annotations for some points
    if len(df) > 0:
        for i in range(min(5, len(df))):
            ax.text(
                price_norm.iloc[i], 
                reviews_norm.iloc[i], 
                df['rating_numeric'].iloc[i], 
                df['title'].iloc[i][:20] + '...',
                fontsize=8
            )
    
    return plt

# ==============================================
# STREAMLIT UI
# ==============================================

def run_streamlit_app():
    """Run the Streamlit app"""
    st.set_page_config(
        page_title="DataWizards",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 Track Your Products")
    st.markdown("### DSAI 103 - DataWizards")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data acquisition method selection
    acquisition_methods = st.sidebar.multiselect(
        "Select Data Acquisition Methods",
        ["BeautifulSoup", "Selenium", "eBay API"],
        default=["BeautifulSoup"]
    )
    
    # Category selection
    selected_categories = st.sidebar.multiselect(
        "Select Categories to Scrape",
        CATEGORIES,
        default=[CATEGORIES[0]]
    )
    
    # Number of items to scrape
    items_per_category = st.sidebar.slider(
        "Items per Category",
        min_value=5,
        max_value=50,
        value=10
    )
    
    # Custom search keyword
    custom_keyword = st.sidebar.text_input("Custom Search Keyword (Optional)")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Data Acquisition", "Network Analysis", "Visualizations", "3D Point Cloud"])
    
    # Tab 1: Data Acquisition
    with tab1:
        st.header("Data Acquisition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_scraping = st.button("Start Data Acquisition", type="primary")
            
        with col2:
            load_existing_data = st.button("Load Existing Data")
        
        if start_scraping:
            # Initialize progress bar
            progress_bar = st.progress(0)
            
            # Container for status messages
            status_container = st.empty()
            
            # Initialize empty lists for products and relationships
            all_products = []
            all_relationships = []
            
            # Get access token if using API
            access_token = None
            if "eBay API" in acquisition_methods:
                status_container.text("Getting eBay API access token...")
                access_token = get_ebay_access_token()
                if not access_token:
                    st.error("Failed to get eBay API access token. API scraping will be skipped.")
            
            # Process each category
            total_steps = len(selected_categories) * len(acquisition_methods)
            current_step = 0
            
            for category in selected_categories:
                # Use custom keyword if provided
                search_term = custom_keyword if custom_keyword else category
                
                # Apply each selected scraping method
                for method in acquisition_methods:
                    current_step += 1
                    progress = current_step / total_steps
                    progress_bar.progress(progress)
                    
                    if method == "BeautifulSoup":
                        status_container.text(f"Scraping {search_term} with BeautifulSoup...")
                        products = soup_scraper(search_term, pages=1)
                        all_products.extend(products)
                        
                    elif method == "Selenium":
                        status_container.text(f"Scraping {search_term} with Selenium...")
                        products = selenium_scraper(search_term, max_items=items_per_category)
                        all_products.extend(products)
                        
                    elif method == "eBay API":
                        status_container.text(f"Fetching {search_term} with eBay API...")
                        if access_token:
                            products = api_scraper(search_term, access_token, max_items=items_per_category)
                            all_products.extend(products)
            
            # Preprocess the scraped data
            status_container.text("Processing product data...")
            processed_products = preprocess_product_data(all_products)
            
            # Scrape relationships data (frequently bought together)
            if len(processed_products) > 0:
                status_container.text("Scraping product relationships...")
                all_relationships = scrape_frequently_bought_together(processed_products)
                all_relationships = create_relationships_data(processed_products, all_relationships)
                
                # Save data to CSV files
                status_container.text("Saving data to CSV files...")
                save_to_csv(processed_products, PRODUCTS_CSV)
                save_to_csv(all_relationships, RELATIONSHIPS_CSV)
                
                # Show summary
                st.session_state["products_data"] = processed_products
                st.session_state["relationships_data"] = all_relationships
                
                status_container.text(f"Completed! Acquired {len(processed_products)} products and {len(all_relationships)} relationships.")
                progress_bar.progress(1.0)
                
                # Show data tables
                st.subheader("Product Data Sample")
                st.dataframe(pd.DataFrame(processed_products))
                
                st.subheader("Relationship Data Sample")
                st.dataframe(pd.DataFrame(all_relationships))
            else:
                status_container.text("No products were found. Please try different categories or methods.")
        
        if load_existing_data:
            try:
                products_df = load_from_csv(PRODUCTS_CSV)
                relationships_df = load_from_csv(RELATIONSHIPS_CSV)
                
                if not products_df.empty:
                    st.session_state["products_data"] = products_df.to_dict('records')
                    st.subheader("Product Data Sample (Loaded)")
                    st.dataframe(products_df)
                else:
                    st.warning(f"No product data found in {PRODUCTS_CSV}")
                
                if not relationships_df.empty:
                    st.session_state["relationships_data"] = relationships_df.to_dict('records')
                    st.subheader("Relationship Data Sample (Loaded)")
                    st.dataframe(relationships_df)
                else:
                    st.warning(f"No relationship data found in {RELATIONSHIPS_CSV}")
            
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    # Tab 2: Network Analysis
    with tab2:
        st.header("Network Analysis")
        
        if "products_data" not in st.session_state or "relationships_data" not in st.session_state:
            st.info("Please acquire or load data first in the 'Data Acquisition' tab")
        else:
            relationships_data = st.session_state.get("relationships_data", [])
            
            if not relationships_data:
                st.warning("No relationship data available for network analysis")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Product Network Visualization")
                    
                    # Create and display network graph
                    with st.spinner("Generating network visualization..."):
                        G, metrics, centrality = perform_network_analysis(relationships_data)
                        if G and metrics and centrality:
                            # Choose centrality measure for visualization
                            centrality_measure = st.selectbox(
                                "Select Centrality Measure for Node Size",
                                ["Degree", "Betweenness"],  # Only these two options
                                index=0
                            )
                            
                            # Create visualization
                            fig = visualize_network(G, centrality[centrality_measure], 
                                                   title=f"Product Network (sized by {centrality_measure} Centrality)")
                            st.pyplot(fig)
                        else:
                            st.warning("Could not generate network visualization")
                
                with col2:
                    st.subheader("Network Metrics")
                    if metrics:
                        for metric, value in metrics.items():
                            st.metric(metric, f"{value:.4f}" if isinstance(value, float) else value)
                    
                    st.subheader("Top Products by Centrality")
                    if centrality:
                        selected_centrality = st.selectbox(
                            "Select Centrality Measure", 
                            ["Degree", "Betweenness"],  # Only these two options
                            key="metric_select"
                        )
                        
                        # Get top 5 products by selected centrality
                        top_products = sorted(
                            centrality[selected_centrality].items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:5]
                        
                        for i, (product, score) in enumerate(top_products, 1):
                            st.write(f"{i}. {product[:30]}... ({score:.4f})")
                    
                    # Community detection
                    st.subheader("Community Analysis")
                    if G:
                        try:
                            from networkx.algorithms import community
                            communities = list(community.greedy_modularity_communities(G))
                            
                            st.write(f"Number of communities: {len(communities)}")
                            
                            # Show top communities by size
                            for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:3], 1):
                                st.write(f"Community {i}: {len(comm)} products")
                        except:
                            st.write("Community detection is not available")
    
    # Tab 3: Visualizations
    with tab3:
        st.header("Data Visualizations")
        
        if "products_data" not in st.session_state:
            st.info("Please acquire or load data first in the 'Data Acquisition' tab")
        else:
            products_data = st.session_state.get("products_data", [])
            
            if not products_data:
                st.warning("No product data available for visualization")
            else:
                # Convert to DataFrame for analysis
                df = pd.DataFrame(products_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Price Distribution by Category")
                    
                    # Create bar chart of average price by category
                    fig, ax = plt.subplots(figsize=(10, 6))
                    category_price = df.groupby('category')['priceValue'].mean().sort_values(ascending=False)
                    category_price.plot(kind='bar', ax=ax)
                    plt.title('Average Price by Category')
                    plt.ylabel('Average Price ($)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Product Count by Category")
                    
                    # Create pie chart of products by category
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
                    plt.title('Products by Category')
                    plt.ylabel('')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Pricing heatmap
                st.subheader("Price Distribution Heatmap")
                
                # Select groupby columns for heatmap
                col1, col2 = st.columns(2)
                with col1:
                    group_col1 = st.selectbox(
                        "Select Primary Dimension",
                        ["category", "source", "condition", "location"],
                        index=0
                    )
                
                with col2:
                    group_col2 = st.selectbox(
                        "Select Secondary Dimension",
                        ["source", "category", "condition", "location"],
                        index=0
                    )
                
                if group_col1 != group_col2:
                    heatmap_fig = create_pricing_heatmap(products_data, [group_col1, group_col2])
                    if heatmap_fig:
                        st.pyplot(heatmap_fig)
                    else:
                        st.warning("Could not generate heatmap with selected dimensions")
                else:
                    st.warning("Please select different dimensions for heatmap")
                
                # Data table with filters
                st.subheader("Interactive Data Table")
                
                # Filter by category
                selected_cat = st.multiselect(
                    "Filter by Category",
                    options=sorted(df['category'].unique()),
                    default=[]
                )
                
                # Filter by price range
                price_range = st.slider(
                    "Price Range",
                    min_value=float(df['priceValue'].min()),
                    max_value=float(df['priceValue'].max()),
                    value=(float(df['priceValue'].min()), float(df['priceValue'].max()))
                )
                
                # Apply filters
                filtered_df = df.copy()
                if selected_cat:
                    filtered_df = filtered_df[filtered_df['category'].isin(selected_cat)]
                
                filtered_df = filtered_df[
                    (filtered_df['priceValue'] >= price_range[0]) & 
                    (filtered_df['priceValue'] <= price_range[1])
                ]
                
                # Show filtered data
                st.dataframe(filtered_df)
    
    # Tab 4: 3D Point Cloud
    with tab4:
        st.header("3D Product Point Cloud")
        
        if "products_data" not in st.session_state:
            st.info("Please acquire or load data first in the 'Data Acquisition' tab")
        else:
            products_data = st.session_state.get("products_data", [])
            
            if not products_data:
                st.warning("No product data available for 3D visualization")
            else:
                st.write("""
                This 3D visualization plots products in a three-dimensional space based on:
                - Price (x-axis)
                - Reviews Count (y-axis)
                - Rating (z-axis)
                
                The size and color of points also encode information about the products.
                """)
                
                cloud_fig = create_point_cloud(products_data)
                if cloud_fig:
                    st.pyplot(cloud_fig)
                else:
                    st.warning("Could not generate 3D point cloud visualization")

# ==============================================
# MAIN ENTRY POINT
# ==============================================

if __name__ == "__main__":
    run_streamlit_app()