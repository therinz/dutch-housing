def convert_elapsed_time(x):
    """Return integer of days since listing online."""

    parts = x.split()
    if len(parts) == 3:
        days = (pd.to_datetime("now") - pd.to_datetime(x)).days
    elif len(parts) == 2:
        time_units = {"weken": 7, "maanden": 40}
        days = int(parts[0].strip("+")) * time_units[parts[1].lower()]
    else:
        days = 1

    return int(days)


def convert_price(x):
    """Return column with integer of price in column."""
    return pd.to_numeric(x.str.extract(r"€ (\S+)", expand=False)
                         .str.replace(".", "")
                         .str.replace(",", ".")
                         .fillna(0))


def build_era(x):
    """Return mean of build time period."""

    if x != x:
        return pd.NA
    start, end = x.split("-")

    return int((int(start) + int(end)) / 2)

drop = ['status', 'acceptance', 'asking_price_original', 'rent_price',
        'rental_agreement', 'rent_price_original', 'sale_type', 'certificates',
        'accessibility', 'prop_extra_type', 'parking_type', 'prop_build_area',
        'opp', 'parking_capacity', 'prop_extra_dimensions', 'ground_area',
        'garden_plaats', 'garden_side', 'garage_size', 'garage_features',
        'garage_isolation', 'kadaster', 'comp_practice', 'company_space',
        'office_space', 'store_space', 'auction_price', 'auction_period',
        'auction_type', 'auction_party']




cols = {'OVE-Vraagprijs': 'asking_price',
        'OVE-Vraagprijs per m²': 'price_m2',
        'OVE-Aangeboden sinds': 'days online',
        'OVE-Status': 'status',
        'OVE-Aanvaarding': 'acceptance',
        'BOU-Soort woonhuis': 'property_type',
        'BOU-Soort bouw': 'new_build',
        'BOU-Bouwjaar': 'build_year',
        'BOU-Soort dak': 'roof_type',
        'OPP-': 'opp',
        'OPP-Perceel': 'property_area',
        'OPP-Inhoud': 'property_volume',
        'IND-Aantal kamers': 'num_rooms',
        'IND-Aantal badkamers': 'num_bathrooms',
        'IND-Badkamervoorzieningen': 'bathroom_features',
        'IND-Aantal woonlagen': 'floors',
        'IND-Voorzieningen':  'features',
        'ENE-Energielabel': 'energy_label',
        'ENE-Isolatie': 'isolation',
        'ENE-Verwarming': 'heating',
        'ENE-Warm water': 'hot_water',
        'ENE-Cv-ketel': 'boiler',
        'KAD-': 'kadaster',
        'BUI-Tuin': 'garden',
        'BUI-Balkon/dakterras': 'balcony',
        'GAR-Soort garage': 'garage_type',
        'GAR-Capaciteit': 'garage_size',
        'GAR-Voorzieningen': 'garage_features',
        'PAR-Soort parkeergelegenheid': 'parking',
        'BUI-Ligging': 'environment',
        'BUI-Voortuin': 'garden_front',
        'BUI-Ligging tuin': 'garden_orientation',
        'OVE-Servicekosten': 'service_fees',
        'BOU-Specifiek': 'specials',
        'BUI-Zonneterras': 'terrace',
        'BER-Schuur/berging': 'storage_type',
        'BER-Voorzieningen': 'storage_features',
        'BUI-Achtertuin': 'garden_back',
        'BER-Isolatie': 'storage_isolation',
        'BOU-Keurmerken': 'certificates',
        'BUI-Patio/atrium': 'garden_patio',
        'OVE-Bijdrage VvE': 'vve_contribution',
        'BOU-Soort appartement': 'apartment_type',
        'BOU-Bouwperiode': 'build_era',
        'IND-Gelegen op': 'building_orientation',
        'VVE-Inschrijving KvK': 'vve_kvk',
        'VVE-Jaarlijkse vergadering': 'vve_am',
        'VVE-Periodieke bijdrage': 'vve_per_contr',
        'VVE-Reservefonds aanwezig': 'vve_reserve_fund',
        'VVE-Onderhoudsplan': 'vve_maintenance',
        'VVE-Opstalverzekering': 'vve_insurance',
        'GAR-Isolatie': 'garage_isolation',
        'BOU-Toegankelijkheid': 'accessibility',
        'OVE-Oorspronkelijke vraagprijs': 'asking_price_original',
        'ENE-Voorlopig energielabel': 'energy_label_temp',
        'OVE-Huurprijs': 'rent_price',
        'OVE-Huurovereenkomst': 'rental_agreement',
        'BUI-Plaats': 'garden_plaats',
        'BUI-Zijtuin': 'garden_side',
        'OVE-Oorspronkelijke huurprijs': 'rent_price_original',
        'BOU-Soort overig aanbod': 'prop_extra_type',
        'BED-Praktijkruimte': 'comp_practice',
        'OVE-Koopmengvorm': 'sale_type',
        'BOU-Soort parkeergelegenheid': 'parking_type',
        'IND-Capaciteit': 'parking_capacity',
        'IND-Afmetingen': 'prop_extra_dimensions',
        'VEI-Prijs': 'auction_price',
        'VEI-Veilingperiode': 'auction_period',
        'VEI-Soort veiling': 'auction_type',
        'VEI-Veilingpartij': 'auction_party',
        'BED-Bedrijfsruimte': 'company_space',
        'BED-Kantoorruimte': 'office_space',
        'BED-Winkelruimte': 'store_space',
        'IND-Perceel': 'ground_area',
        'BOU-Soort object': 'prop_build_area'}

df[['OVE-Status', 'OVE-Aanvaarding', 'BOU-Soort bouw']]