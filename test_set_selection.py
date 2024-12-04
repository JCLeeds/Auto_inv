import pandas as pd 
from global_land_mask import globe

def LiCS_test_set_selection(file_path,start_date,end_date,mag_lower,mag_upper):
    # Define column names
    columns = [
        'ID', 'Magnitude', 'Depth', 'Timestamp', 'Latitude', 'Longitude', 'Link', 'Location_and_Country','On_Land'
    ]

    # Initialize an empty list to store rows
    data = []

    # Read and process the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into fields
            parts = line.strip().split('\t')
            
            # Extract data
            if len(parts) == 9:
                id_, magnitude, depth, timestamp, latitude, longitude, link, location, country = parts
                # Remove HTML tags from link
                link = link.split('href=')[1].split(' ')[0].strip("'")
                
                # Combine location and country
                location_and_country = f"{location} {country}"

                on_land = globe.is_land(float(latitude), float(longitude))
                
                # Append data to the list
                data.append([
                    id_,
                    float(magnitude),
                    float(depth),
                    timestamp,
                    float(latitude),
                    float(longitude),
                    link,
                    location_and_country,
                    on_land
                ])
            else:
                print(f"Skipping malformed line: {line}")

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Filter DataFrame for events between 2021 and 2023
    # start_date = '2021-01-01'
    # end_date = '2024-08-30'
    filtered_df = df[
    (df['Timestamp'] >= start_date) & 
    (df['Timestamp'] <= end_date) & 
    (df['On_Land'] == True) &
    (df['ID'].str.startswith('us')) &
    (df['Magnitude'] >= mag_lower) &
    (df['Magnitude'] <= mag_upper)]

    print(len(filtered_df))

    # Display the DataFrame
    # print(df)

    print(filtered_df)
    # Optional: Save DataFrame to CSV file
    filtered_df.to_csv('2021_2024_08_30_on_land_usIds_5_5_to_6_5.csv', index=False)
    return filtered_df


if __name__ == '__main__':
    LiCS_test_set_selection('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/LiCS_EQ_Responder.csv','2021-01-01','2024-08-30',5.5,6.5)