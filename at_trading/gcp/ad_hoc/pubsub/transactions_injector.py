#!/usr/bin/env python

import datetime, json, os, random, time

# Set the `project` variable to a Google Cloud project ID.
project = 'at-ml-platform'

FIRST_NAMES = ['Monet', 'Julia', 'Angelique', 'Stephane', 'Allan', 'Ulrike', 'Vella', 'Melia',
               'Noel', 'Terrence', 'Leigh', 'Rubin', 'Tanja', 'Shirlene', 'Deidre', 'Dorthy', 'Leighann',
               'Mamie', 'Gabriella', 'Tanika', 'Kennith', 'Merilyn', 'Tonda', 'Adolfo', 'Von', 'Agnus',
               'Kieth', 'Lisette', 'Hui', 'Lilliana', ]
CITIES = ['Washington', 'Springfield', 'Franklin', 'Greenville', 'Bristol', 'Fairview', 'Salem',
          'Madison', 'Georgetown', 'Arlington', 'Ashland', ]
STATES = ['MO', 'SC', 'IN', 'CA', 'IA', 'DE', 'ID', 'AK', 'NE', 'VA', 'PR', 'IL', 'ND', 'OK', 'VT', 'DC', 'CO', 'MS',
          'CT', 'ME', 'MN', 'NV', 'HI', 'MT', 'PA', 'SD', 'WA', 'NJ', 'NC', 'WV', 'AL', 'AR', 'FL', 'NM', 'KY', 'GA',
          'MA',
          'KS', 'VI', 'MI', 'UT', 'AZ', 'WI', 'RI', 'NY', 'TN', 'OH', 'TX', 'AS', 'MD', 'OR', 'MP', 'LA', 'WY', 'GU',
          'NH']
PRODUCTS = ['Product 2', 'Product 2 XL', 'Product 3', 'Product 3 XL', 'Product 4', 'Product 4 XL', 'Product 5',
            'Product 5 XL', ]

while True:
    first_name, last_name = random.sample(FIRST_NAMES, 2)
    data = {
        'tr_time_str': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'first_name': first_name,
        'last_name': last_name,
        'city': random.choice(CITIES),
        'state': random.choice(STATES),
        'product': random.choice(PRODUCTS),
        'amount': float(random.randrange(50000, 70000)) / 100,
    }

    # For a more complete example on how to publish messages in Pub/Sub.
    #   https://cloud.google.com/pubsub/docs/publisher
    message = json.dumps(data)
    command = "gcloud --project={} pubsub topics publish transactions --message='{}'".format(project, message)
    print(command)
    os.system(command)
    time.sleep(random.randrange(1, 5))
