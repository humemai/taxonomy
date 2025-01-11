# Wikidata

This repository contains Python scripts designed to process and analyze the raw Wikidata
dump files. It enables the extraction of key information, such as entity labels,
relationships, and property usage statistics. There are five scripts, and they can all
be run in parallel. As of 05-Jan-2025, there are 113,472,282 entities and 12,327
properties in wikidata the database.

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

   As of 05-Jan-2025, the file size for `latest-all.json.gz` is approximately `142 GB`.

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

**On my machine, this took about 1 hour and 11 minutes.**

## Get subclass_of (P279) triples: [`run_p279.py`](run_p279.py)

To extract `subclass_of` (`P279`) relationships from the Wikidata dump, you can use the
provided Python script `run_p279.py`. This script processes the raw JSON dump, extracts
`P279` claims, and saves them in a TSV format for easier downstream analysis.

### Steps to Extract `P279` Triples

1. **Run the `run_p279.py` script**: Use the following command to extract `P279`
   triples:

   ```bash
   python run_p279.py --dump_file latest-all.json.gz --p279_dir P279
   --num_entities_per_batch 50000 --dummy
   ```

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

**On my machine, this took about 13 hours and 56 minutes.**

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

**On my machine, this took about 15 hours and 58 minutes.**

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

**On my machine, this took about 2 hours and 38 minutes.**

## Get the stats of the properties used: [`run_property_stats.py`](run_property_stats.py)

The `run_property_stats.py` script calculates the usage statistics of properties in the
Wikidata JSON dump and saves the results as a JSON file.

### Steps to Get Property Stats

1. Run the script:

   python run_property_stats.py --dump_file latest-all.json.gz --output_file
   property_stats.json --dummy

   - `--dump_file`: Path to the Wikidata JSON dump (`latest-all.json.gz`).
   - `--output_file`: Path to save the `property_stats.json` file (default:
     `property_stats.json`).
   - `--dummy`: Optional flag to process only the first 10,000 entities for testing.

2. Output Structure: The property statistics are saved as a JSON file
   (`property_stats.json`) with the following format:

   ```json
   { "P31": 25000, "P279": 18000, ... }
   ```

   Each property ID (e.g., `P31`) is a key, and its value is the number of times the
   property appears in the dump.

3. Logs: A log file `run_property_stats.log` is generated, containing:
   - Total processing time (in days, hours, minutes, and seconds).
   - Total entities processed.
   - Decoding errors.
   - Path to the output file.
   - Top 5 most frequently used properties.

**On my machine, this took about 2 hours and 40 minutes.**

## Building a hierarchy using the P31 and P279 triples

### [`process_p31_p279.py`](./process_p31_p279.py) Script Overview

The [`process_p31_p279.py`](./process_p31_p279.py) script is designed to process
subclass (P279) and instance (P31) relationships from TSV files to analyze class
distributions within a knowledge graph. Additionally, it processes property usage
statistics. The script performs the following key tasks:

#### 1. Directory Creation

- **Function:** `create_output_directory`
- **Purpose:** Ensures that a directory named `process_p31_p279` exists. If it doesn't,
  the function creates it. All output files, including JSONs and figures, are saved
  within this directory.

#### 2. Property Processing

- **Loading Data:**
  - **Property Usage Statistics:** Loaded from `property_stats.json` using the
    `load_properties_used` function.
  - **Property Labels:** Loaded from `properties.json` using the
    `load_properties_labels` function.
- **Data Analysis:**
  - Computes cumulative distribution of property usage.
  - Identifies the number of properties required to cover specified thresholds (80%,
    90%, 95%, 99%).
- **Visualization:**
  - Plots and saves the cumulative distribution of properties using
    `plot_cumulative_distribution_properties`.
- **Reporting:**
  - Prints details of the top 100 properties with their labels and usage counts using
    `print_top_properties`.

#### 3. P31/P279 Processing

- **Loading Data:**
  - **Entity Labels:** Loaded from `entityid2label.json` using the `load_entity_labels`
    function.
  - **Subclass Relationships (P279):** Loaded from TSV files in the `./P279/` directory
    using `load_relationships_p31_p279`.
  - **Instance Relationships (P31):** Loaded from TSV files in the `./P31/` directory
    using `load_relationships_p31_p279`.
- **Data Analysis:**
  - Counts the frequency of each class based on instance data using
    `count_classes_p31_p279`.
  - Computes cumulative distribution of class counts using
    `compute_cumulative_distribution_classes`.
  - Identifies the number of classes required to cover specified thresholds (80%, 90%,
    95%, 99%) using `find_thresholds_classes`.
- **Visualization:**
  - Plots and saves the cumulative distribution of classes using
    `plot_cumulative_distribution_classes`.
- **Reporting:**
  - Prints details of the top 100 classes with their labels and counts using
    `print_top_classes`.

#### 4. Data Saving

- **JSON Files:**
  - Saves `class_counts` and `child_to_parents` as individual JSON files within the
    `process_p31_p279` directory.
- **Figures:**
  - Saves generated plots (`properties_cumulative_distribution.png` and
    `classes_cumulative_distribution.png`) within the `process_p31_p279` directory.

#### 7. Usage Instructions

- **Dependencies:**
  - Python 3.10+
  - matplotlib
  - tqdm
- **Installation:**
  - Install missing packages using `pip`:
    ```bash
    pip install matplotlib tqdm
    ```
- **Execution:**
  ```bash
  python process_p31_p279.py
  ```

#### 8. Outputs

- **JSON Files:**
  - `process_p31_p279/class_counts.json`
  - `process_p31_p279/child_to_parents.json`
- **Figures:**
  - `process_p31_p279/properties_cumulative_distribution.png`
  - `process_p31_p279/classes_cumulative_distribution.png`

**On my machine, this took about 6 minutes.**

### [`get_paths.py`](./get_paths.py) Script Overview

[`get_paths.py`](./get_paths.py) is a Python script designed to
generate and export unique hierarchical paths for the top N classes from a dataset. It
processes relationships between entities, extracts meaningful paths based on specified
directions, and logs essential statistics for each class. The output consists of
separate TSV (Tab-Separated Values) files for each class along with corresponding log
files.

#### Key Features

- **Path Generation:** Constructs unique upward and/or downward paths using Depth-First
  Search (DFS) based on user-specified directions.
- **Trie Data Structure:** Utilizes a Trie to efficiently store and retrieve unique
  paths, eliminating duplicates.
- **Configurable Parameters:** Users can specify the number of top classes (`N`),
  minimum path depth (`min_depth), maximum path depth (`max_depth`), maximum paths per
class (`max_paths_per_class`), and the direction of path generation (`upward`,
`downward`, or `both`).
- **Progress Monitoring:** Integrates `tqdm` for real-time progress tracking.
- **Logging:** Records statistics such as the number of paths generated, unique paths
  extracted, time taken, and memory usage for each class.
- **Error Handling:** Manages file I/O errors and detects cycles during path generation
  to prevent infinite loops.

#### Usage

Run the script via the command line with required and optional arguments:

```bash
python get_paths.py --num_classes 20 --max_depth 5 --max_paths_per_class 1000
--batch_size 50000 --direction both
```

- `--num_classes`: Number of top classes to process (default: 10)
- `--min_depth`: Minimum depth for path generation (default: None)
- `--max_depth`: Maximum depth for path generation (default: None)
- `--max_paths_per_class`: Maximum number of paths per class for each direction
  (default: None)
- `--batch_size`: Number of combined paths per batch TSV file (default: 50000)
- `--direction`: Direction of paths to include (`upward`, `downward`, or `both`)
  **(required)**

#### Core Components

- **TrieNode & PathTrie Classes:** Implement a Trie for storing unique paths. `TrieNode`
  represents each node, while `PathTrie` handles insertion and traversal of paths.
- **Functions:**
  - `get_memory_usage()`: Returns current memory usage in MB.
  - `generate_paths_dfs()`: Generates all complete paths from a node using DFS.
  - `sample_and_combine_paths()`: Processes top `num_classes` classes, generates paths,
    ensures uniqueness, and exports them based on the specified direction.
  - `invert_mapping()`: Converts child-to-parent mappings to parent-to-child mappings.
  - `format_time()`: Formats elapsed time into a readable string.

#### Execution Flow

1. **Argument Parsing:** Handles user-specified parameters, including the required
   `--direction`.
2. **Setup:**
   - Creates the output directory (`extracted_paths` by default).
   - Loads `class_counts.json` and `child_to_parents.json` from the
     `./process_p31_p279/` directory.
   - Inverts the child-to-parent mapping to parent-to-children.
3. **Path Processing:**
   - For each of the top `num_classes` classes:
     - Generates upward and/or downward paths based on the `--direction` argument with
       optional depth and path limits.
     - Samples paths if `max_paths_per_class` is specified.
     - Inserts paths into a Trie to ensure uniqueness.
     - Depending on the direction:
       - **Both Directions (`both`)**: Combines unique upward and downward paths and
         exports them in batched TSV files.
       - **Only Upward (`upward`)**: Exports unique upward paths in batched TSV files.
       - **Only Downward (`downward`)**: Exports unique downward paths in batched TSV
         files.
     - Logs statistics to `{class}.log` including path counts, time taken, and memory
       usage.
4. **Completion:** Reports total execution time and confirms successful completion.

#### Dependencies

- **Standard Libraries:** `os`, `json`, `random`, `time`, `argparse`, `itertools`,
  `collections`, `typing`, `sys`.
- **Third-Party Libraries:** `tqdm` (progress bars), `psutil` (memory monitoring).

Ensure all dependencies are installed, for example:

```bash
pip install tqdm psutil
```

#### Performance Considerations

- **Memory Efficiency:** Uses Tries to minimize memory usage by avoiding duplicate path
  storage.
- **Sampling & Depth Limitation:** Controls resource usage by limiting the number of
  paths and their depth.
- **Progress Tracking:** Provides real-time feedback to monitor processing status.

#### Output

- **TSV Files:** For each processed class, one or more TSV files are created:
  - `batch_1.tsv`, `batch_2.tsv`, etc.: Each file contains a batch of unique paths based
    on the specified direction.
- **Log Files:** For each processed class, a log file `{class}.log` is created
  containing:
  - Initial number of upward and/or downward paths.
  - Number of unique upward and/or downward paths after Trie insertion.
  - Time taken for processing the class.
  - Memory usage during processing.
  - Direction of paths included.

#### Error Handling

- **File I/O Errors:** The script handles errors related to reading input files and
  writing output files, logging appropriate error messages without terminating the
  entire process.
- **Cycle Detection:** During path generation, the script detects and skips paths that
  would introduce cycles, preventing infinite loops.

#### Example

To process the top 15 classes with a maximum path depth of 7, limit to 500 paths per
class in each direction, and include both upward and downward paths, run:

```bash
python get_paths.py --num_classes 15 --min_depth 4 --max_depth 7 --max_paths_per_class 500
--direction both
```

This command will generate unique upward and downward paths for the top 15 classes, each
path not exceeding 7 entities in length, and no more than 500 paths per direction per
class. The results will be saved in the `extracted_paths` directory with corresponding
log files.

#### Notes

- Ensure that the input JSON files (`class_counts.json` and `child_to_parents.json`) are
  correctly formatted and located in the `./process_p31_p279/` directory.
- The script is optimized for large datasets, but resource usage can still be
  significant depending on the size and complexity of the input data. Adjust
  `min_depth`, `max_depth` and `max_paths_per_class` as needed to balance performance
  and comprehensiveness.

This script is ideal for analyzing hierarchical data structures, such as taxonomies or
ontologies, by extracting and managing unique paths efficiently.

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
