import pandas as pd
import numpy as np
import requests
from src import config
import json
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

from sklearn.cluster import KMeans
from joblib import dump, load
import os
from src import geopy_utils
import shutil

import folium

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

def getNearbyVenues(names, latitudes, longitudes, radius=1000, LIMIT=40):
    '''Gathers nearby recommended venues using the Foursquare API. 
    '''
    # Empty list to append venues to
    venues_list = []
    for name, lat, lng in zip(names, latitudes, longitudes):
        print("Gathering venues in ", name)

        # Specifies endpoint
        url = "https://api.foursquare.com/v2/search/recommendations"

        params = dict(
            client_id=config.CLIENT_ID,
            client_secret=config.CLIENT_SECRET,
            v='20191129',
            ll=(str(lat) + "," + str(lng)),
            radius=radius,
            limit=LIMIT)

        # Makes request
        results = requests.get(url=url, params=params).json()["response"]['group']['results']
        

        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng,            
            v['venue']['name'], 
            v['venue']['id'],
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'], 
            v['venue']['categories'][0]['name']) for v in results])
        

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['City', 
                             'City Latitude', 
                             'City Longitude', 
                             'Venue', 
                             'id',
                             'Venue Latitude', 
                             'Venue Longitude', 
                             'Venue Category']
    
    return nearby_venues


def get_rating(row):
    '''
    Allows to get the rating given a dataframe with venue id's
    Mostly applicable in the .apply() method 
    '''
    try:
        venue_id = row['id']
    except:
        venue_id = row['venue.id']
    if len(venue_id) == 0:
        return None
    else:
        url = "https://api.foursquare.com/v2/venues/{}".format(venue_id)

        params = dict(
          client_id=config.CLIENT_ID,
          client_secret=config.CLIENT_SECRET,
          v='20191129')

        resp_rat = requests.get(url=url, params=params)
        data_rat = json.loads(resp_rat.text)

        rating = json_normalize(data_rat)

        rating = rating["response.venue.rating"][0]

        return rating


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


def return_most_common_venues(row, num_top_venues):
    '''Returns most common venues'''
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


def get_cities_common_venues(final_grouped, num_top_venues=10, saving_dir=""):
    indicators = ['st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th', 'th', 'th']

    # create columns according to number of top venues
    columns = ['City']
    for ind in np.arange(num_top_venues):
        try:
            columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
        except:
            columns.append('{}th Most Common Venue'.format(ind+1))

    # create a new dataframe with the same neighborhoods as in toronto_grouped
    neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
    neighborhoods_venues_sorted['City'] = final_grouped['City']

    # Fill each row on each column.
    for ind in np.arange(final_grouped.shape[0]):
        neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(final_grouped.drop(columns=["Rating"]).iloc[ind, :], num_top_venues)

    neighborhoods_venues_sorted.to_csv(os.path.join(saving_dir, "CSV_Files/most_common_venues.csv"), index=False)
    return neighborhoods_venues_sorted

def prepare_data_for_kmeans(final_grouped, norm="std"):
    final_grouped_clustering = final_grouped.drop('City', 1)
    
    x = final_grouped_clustering #returns a numpy array

    if norm == "minmax":
        # Instantiating the scalers
        scaler = preprocessing.MinMaxScaler()
    elif norm == "std":
        scaler = preprocessing.StandardScaler()
    else:
        scaler = preprocessing.MinMaxScaler()

    # Fit and transform the dataframe
    x_scaled = scaler.fit_transform(x)

    # Convert fitted values into a DataFrame
    final_grouped_clustering_norm = pd.DataFrame(x_scaled, columns=final_grouped_clustering.columns)
    
    return final_grouped_clustering_norm


def generate_folium_map(num_clusters, final_df, saving_dir=""):
    # create map
    latitude = 39.7392
    longitude = -104.9903
    map_clusters = folium.Map(location=[latitude, longitude], zoom_start=5)

    # set color scheme for the clusters
    x = np.arange(num_clusters)
    ys = [i + x + (i*x)**2 for i in range(num_clusters)]
    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    rainbow = [colors.rgb2hex(i) for i in colors_array]

    # add markers to the map
    markers_colors = []
    for lat, lon, poi, cluster in zip(final_df['Venue Latitude'], final_df['Venue Longitude'], final_df['City'], final_df['Cluster Labels']):
        label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
        folium.CircleMarker(
            [lat, lon],
            radius=5,
            popup=label,
            color=rainbow[int(cluster)-1],
            fill=True,
            fill_color=rainbow[int(cluster)-1],
            fill_opacity=0.7).add_to(map_clusters)

    map_clusters.save(os.path.join(saving_dir, "Folium/clustering_results_map.html"))
    
    return map_clusters

               
def get_analysis_per_city(city_state_list, num_venues=20, radius=1000, clusters=3, saving_dir=None):
    """This function

    Notice that the num_venues argument will specified the number of regular and premium calls for each city.


    Args:
        city_state_list (list): a list containing "city, state" strings for analysis.
        num_venues (int, optional): The number of recommended venues to get. Defaults to 20.
        radius (int, optional): The radius in meters for which to get recommended venues. Defaults to 1000.
        clusters (int, optional): The number of clusters to fit. Must be at most the number of cities queried. Defaults to 3.
        saving_dir (str, optional): name of the directory to be created where the results will be stored. Defaults to None.

    Returns:
        None
    """    
    
    if os.path.exists(saving_dir):
        print("Directory exists. Re-initializing...")
        shutil.rmtree(saving_dir)
        os.makedirs(saving_dir)
        os.makedirs(os.path.join(saving_dir, "CSV_Files/"))
        os.makedirs(os.path.join(saving_dir, "KNN_Model/"))
        os.makedirs(os.path.join(saving_dir, "Folium/"))
    else:
        print("Directory does not exists. Initializing...")
        os.makedirs(saving_dir)
        os.makedirs(os.path.join(saving_dir, "CSV_Files/"))
        os.makedirs(os.path.join(saving_dir, "KNN_Model/"))
        os.makedirs(os.path.join(saving_dir, "Folium/"))

    with open(os.path.join(saving_dir, "cities.txt"), "w") as output:
        for i in city_state_list:
            output.write(i + "\n")

    cities_coordinates = geopy_utils.get_coordinates_df(city_state_list)    
    cities_coordinates["Cities"] = cities_coordinates["Cities"].apply(lambda x: x.split(",")[0])
    
    final_df = getNearbyVenues(names = cities_coordinates['Cities'],
                               latitudes = cities_coordinates['Latitude'],
                               longitudes = cities_coordinates['Longitude'],
                               LIMIT = num_venues,
                               radius = radius)
    
    final_df['Venue Rating'] = final_df.apply(get_rating, axis=1)


    city_venues_csv_name = os.path.join(saving_dir, "CSV_Files/cities_venues.csv")
    final_df.to_csv(city_venues_csv_name, index=False)
    final_df = pd.read_csv(city_venues_csv_name)
    
    print('There are {} uniques venue categories.'.format(len(final_df['Venue Category'].unique())))
    final_df["Venue Category"].value_counts(normalize=True).head(10).plot(kind="barh")
    plt.xlabel('Fraction')
    plt.savefig(os.path.join(saving_dir, "venue_distribution.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # one hot encoding
    venue_cat_onehot = pd.get_dummies(final_df[['Venue Category']], prefix="", prefix_sep="")

    # add neighborhood column back to dataframe
    venue_cat_onehot['City'] = final_df['City'] 
    venue_cat_onehot['Rating'] = final_df['Venue Rating'] 

    final_grouped = venue_cat_onehot.groupby('City').mean().reset_index()
    final_grouped.Rating = MinMaxScaler().fit_transform(final_grouped[["Rating"]])
    
    neighborhoods_venues_sorted = get_cities_common_venues(final_grouped=final_grouped, saving_dir=saving_dir)
    
    final_grouped_clustering_norm = prepare_data_for_kmeans(final_grouped)
    
    # run k-means clustering
    kmeans_model = KMeans(n_clusters=clusters, n_init=20, max_iter=500, random_state=100).fit(
        final_grouped_clustering_norm)
    
    dump(kmeans_model, os.path.join(saving_dir, 'KNN_Model/knn.joblib'))

    # add clustering labels
    neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans_model.labels_)

    # merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
    final_merged = final_df.join(neighborhoods_venues_sorted.set_index('City'), on='City')
    final_merged.to_csv(os.path.join(saving_dir, "CSV_Files/clustering_results.csv"), index=False)

    
    map_clusters = generate_folium_map(clusters, final_merged, saving_dir=saving_dir)

    print("Finished")

    return None





# testing, model = get_analysis_per_city(cities_list)

# testing.head()
