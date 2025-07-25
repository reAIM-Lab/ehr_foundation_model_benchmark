## Phenotypes

We curated 11 phenotypes as downstream tasks. Each phenotype contains two json files. One for defining the case cohort, another for defining the at risk cohort.

To use the json files, there are two approaches depending on if a local OHDSI Atlas instance is available.

1. If local OHDSI Atlas instance exist, you can load the phenotype json files into your cohort definition section and generate the cohorts directly.
2. If local OHDSI Atlas instance doesn't exist, you can use this [CohortGenerator R package](https://github.com/OHDSI/CohortGenerator?tab=readme-ov-file) to execute the json files and generate cohort.  