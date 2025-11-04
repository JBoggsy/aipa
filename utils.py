import geocoder


def get_geolocation() -> dict:
    """
    Get the geolocation based on the user's IP address.

    Returns:
        dict: A dictionary containing geolocation information:
            - lat (float): Latitude
            - lng (float): Longitude
            - city (str): City name
            - state (str): State name
            - country (str): Country name
    """
    g = geocoder.ip('me')
    return {
        'lat': g.latlng[0],
        'lng': g.latlng[1],
        'city': g.city,
        'state': g.state,
        'country': g.country
    }