def preprocess_df(df, title):
    if title == "Heart Failure Clinical Records Dataset":
        dtype_dict = {
            "age": "float64",
            "anaemia": "category",
            "creatinine_phosphokinase": "float64",
            "diabetes": "category",
            "ejection_fraction": "float64",
            "high_blood_pressure": "category",
            "platelets": "float64",
            "serum_creatinine": "float64",
            "serum_sodium": "float64",
            "sex": "category",
            "smoking": "category",
            "DEATH_EVENT": "category",
        }
    elif title == "Covid19":
        # Specify data types for all columns
        dtype_dict = {
            "sex": "category",
            "age": "category",
            "country": "category",
            "province": "category",
            "city": "category",
            "infection_case": "category",
            "infection_order": "category",
            "elementary_school_count": "category",
            "kindergarten_count": "category",
            "university_count": "category",
            "academy_ratio": "category",
            "elderly_population_ratio": "category",
            "elderly_alone_ratio": "category",
            "nursing_home_count": "category",
            "avg_temp": "category",
            "min_temp": "category",
            "max_temp": "category",
            "precipitation": "category",
            "max_wind_speed": "category",
            "most_wind_direction": "category",
            "avg_relative_humidity": "category",
            "label": "category",
        }

    # Apply the data types
    for column, dtype in dtype_dict.items():
        df[column] = df[column].astype(dtype)

    return df, dtype_dict
