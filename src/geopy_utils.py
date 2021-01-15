import pandas as pd
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values


# Instantiating the geographic locator
geolocator = Nominatim(user_agent="ny_explorer")


def get_coordinates_df(cities):
    '''Allows the user to create a dataframe which contains
    the latitude and longitude of the requested cities.

    cities [list]: a list containing city and state. For example:
    ['San Diego, California', 'Seattle, Washington']
    '''
    # Initiating empty dataframe
    cities_location = pd.DataFrame(columns=["Cities", "Latitude", "Longitude"])
    
    print("Gathering location information...")
    # Looping through the list of cities
    for i in cities:
        # Specifying the address
        address = "Downtown " + i 

        # Gathering the location coordinates
        location = geolocator.geocode(address)

        # Placing coordinates to variables
        latitude = location.latitude
        longitude = location.longitude

        # Creating a dataframe with calculated values
        processing = pd.DataFrame({"Cities":[i],
                                "Latitude":[latitude],
                                "Longitude":[longitude]})
        
        # Appending Values to cities_location dataframe
        cities_location = cities_location.append(processing)

    print("Finished.")
    return cities_location