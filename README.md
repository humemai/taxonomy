# Wikidata

This repository contains Python scripts designed to process and analyze the raw Wikidata
dump files. It enables the extraction of key information, such as entity labels,
relationships, and property usage statistics. There are four scripts, and they can be
run in parallel.

## Storage Requirements

Working with Wikidata requires significant storage space. Ensure you have sufficient
capacity before proceeding.

### Downloading the Latest Dump File

Wikidata provides entity dumps in various formats. The recommended file for this process
is `latest-all.json.gz`.

1. **Install aria2 for faster downloads**:

   ```sh
   pip install aria2
   ```

2. **Download the file**:

   ```sh
   aria2c --max-connection-per-server=16 https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz
   ```

   As of 18-Dec-2024, the file size for `latest-all.json.gz` is approximately `141 GB`.

## Understanding Wikidata Entities

### What Does a Wikidata Entity Look Like?

A typical Wikidata entity is structured as a JSON object. It includes various attributes
like `id`, `type`, `labels`, `descriptions`, `aliases`, and `claims`. Below is an
example of a Wikidata entity:

Example Entity (`Q42` - Douglas Adams)

```json
{
  "id": "Q42",
  "type": "item",
  "labels": {
    "en": {
      "language": "en",
      "value": "Douglas Adams"
    }
  },
  "descriptions": {
    "en": {
      "language": "en",
      "value": "English author and humorist"
    }
  },
  "aliases": {
    "en": [
      {
        "language": "en",
        "value": "Douglas Noel Adams"
      }
    ]
  },
  "claims": {
    "P31": [
      {
        "mainsnak": {
          "snaktype": "value",
          "property": "P31",
          "datavalue": {
            "value": {
              "entity-type": "item",
              "numeric-id": 5,
              "id": "Q5"
            },
            "type": "wikibase-entityid"
          },
          "datatype": "wikibase-item"
        },
        "type": "statement",
        "id": "Q42$F8F97AE9-C88E-41FA-B527-F34A49EB2E59",
        "rank": "normal"
      }
    ]
  },
  "modified": "2024-12-03T18:56:58Z"
}
```

## Get all the properties as json: [`run_properties.py`](run_properties.py)

Simply run:

```bash
python run_properties.py
```

This will run the SPARQL query to fetch properties from the Wikidata website, retrieve
their aliases and descriptions, and save the results to a file called `properties.json`.

`properties.json` will look like:

```json
{
  "P6": {
  "label": "head of government",
  "aliases": ["leader", "chief of state"],
  "description": "The principal leader of a government, particularly a nation."
},
...
}
```

Each entry in the JSON file contains the following fields:

- `property_id`: The unique identifier for the property (e.g., `P6`).
- `label`: The name or label of the property (e.g., "head of government").
- `aliases`: A list of alternative labels for the property (e.g., ["leader", "chief of
  state"]).
- `description`: A description of the property (e.g., "The principal leader of a
  government, particularly a nation.").

## Get subclass_of (P279) triples: [`run_p279.py`](run_p279.py)

To extract `subclass_of` (`P279`) relationships from the Wikidata dump, you can use the
provided Python script `run_p279.py`. This script processes the raw JSON dump, extracts
`P279` claims, and saves them in a TSV format for easier downstream analysis.

### Steps to Extract `P279` Triples

1. **Run the `run_p279.py` script**: Use the following command to extract `P279`
   triples:

   python run_p279.py --dump_file latest-all.json.gz --p279_dir P279
   --num_entities_per_batch 50000 --dummy

   - `--dump_file`: Path to the Wikidata JSON dump (`latest-all.json.gz`).
   - `--p279_dir`: Directory where the `P279` triples will be stored (default: `P279`).
   - `--num_entities_per_batch`: Number of entities to process per batch (default:
     50,000).
   - `--dummy`: Optional flag to process only the first batch for testing purposes.

2. **Output Structure**: The script will save `P279` triples in a tab-separated file
   (`.tsv`) with the following format:

   ```tsv
   entity_id\tproperty_id\tvalue_id
   ...
   ```

   Each file will contain a batch of triples, named as `batch_0.tsv`, `batch_1.tsv`,
   etc., in the specified output directory.

3. **Check Logs**: A log file, `run_p279.log`, will be created in the parent directory
   of `--p279_dir`. It includes information about the processing time, total entities
   processed, and any errors encountered during JSON decoding.

4. **Sample Output**: Suppose you extract `P279` triples from the provided example
   entity (`Q42` - Douglas Adams). The result in the output file would look like this:

   ```tsv
   entity_id\tproperty_id\tvalue_id
   Q42\tP279\tQ5
   Q123\tP279\tQ456
   ...
   ```

### Tips for Efficient Processing

- **Batch Size**: Adjust the `--num_entities_per_batch` parameter based on your system’s
  memory and processing capabilities. Larger batches reduce file overhead but require
  more memory.
- **Dummy Mode**: Use `--dummy` mode to test the script with minimal processing.
- **Parallel Downloads**: For faster download of the dump file, use a tool like `aria2`
  as mentioned earlier.

Once the script has finished running, you will have all `P279` triples extracted and
saved in an easy-to-process format, ready for further analysis or integration into other
systems.

## Get instance_of (P31) triples: [`run_p31.py`](run_p31.py)

To extract `instance_of` (`P31`) relationships from the Wikidata dump, use the
`run_p31.py` script:

### Steps to Extract `P31` Triples

1. Run the script: `python run_p31.py --dump_file latest-all.json.gz --p31_dir P31
--num_entities_per_batch 50000`

   - `--dump_file`: Path to the Wikidata JSON dump (`latest-all.json.gz`).
   - `--p31_dir`: Directory to store `P31` triples (default: `P31`).
   - `--num_entities_per_batch`: Number of entities per batch (default: 50,000).
   - `--dummy`: Optional flag to process only the first batch.

2. Output Structure: The script saves triples in `.tsv` files under `--p31_dir`:

   ```tsv
   entity_id\tproperty_id\tvalue_id
   Q42\tP31\tQ5
   ...
   ```

3. Logs: A log file `run_p31.log` is created with details like processing time and
   errors.

4. Note that this can run in parallel with `run_p279.py`. Do so to save time.

## Get English labels: [`run_entityid2label.py`](run_entityid2label.py)

The `run_entityid2label.py` script extracts `entity ID` to `English labels` mappings
from the Wikidata JSON dump and saves them as a JSON file.

### Steps to Extract English Labels

1. Run the script:

   python run_entityid2label.py --dump_file latest-all.json.gz --output_file
   entityid2label.json --dummy

   - `--dump_file`: Path to the Wikidata JSON dump (`latest-all.json.gz`).
   - `--output_file`: Path to save the `entityid2label.json` file (default:
     `entityid2label.json`).
   - `--dummy`: Optional flag to process only the first 10,000 entities for testing.

2. Output Structure: The extracted labels are saved as a JSON file
   (`entityid2label.json`) with the following format:

   ```json
   { "Q42": "Douglas Adams", "Q123": "September", ... }
   ```

3. Logs: A log file `run_entityid2label.log` is generated, containing:
   - Total processing time (in days, hours, minutes, and seconds).
   - Total entities processed.
   - Total entities with English labels.
   - Decoding errors.
   - Path to the output file.

This script is ideal for extracting and saving mappings of entity IDs to their English
labels in a simple, usable format.

## Get the stats of the properties used: [`run_properties_stats.py`](run_properties_stats.py)

The `run_properties_stats.py` script calculates the usage statistics of properties in
the Wikidata JSON dump and saves the results as a JSON file.

### Steps to Get Property Stats

1. Run the script:

   python run_properties_stats.py --dump_file latest-all.json.gz --output_file
   properties_stats.json --dummy

   - `--dump_file`: Path to the Wikidata JSON dump (`latest-all.json.gz`).
   - `--output_file`: Path to save the `properties_stats.json` file (default:
     `properties_stats.json`).
   - `--dummy`: Optional flag to process only the first 10,000 entities for testing.

2. Output Structure: The property statistics are saved as a JSON file
   (`properties_stats.json`) with the following format:

   ```json
   { "P31": 25000, "P279": 18000, ... }
   ```

   Each property ID (e.g., `P31`) is a key, and its value is the number of times the
   property appears in the dump.

3. Logs: A log file `run_properties_stats.log` is generated, containing:
   - Total processing time (in days, hours, minutes, and seconds).
   - Total entities processed.
   - Decoding errors.
   - Path to the output file.
   - Top 5 most frequently used properties.

## Contributing

Contributions are what make the open source community such an amazing place to be learn,
inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory, to ensure
   code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Authors

- [Taewoon Kim](https://taewoon.kim/)
