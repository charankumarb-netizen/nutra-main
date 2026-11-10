# NutriCheck - Person 2: Dataset

## Overview

This module is responsible for validating and analyzing the medical transcription dataset used by the NutriCheck project.

The dataset is already preprocessed, so this module does **not** perform aggressive text cleaning or modify the original medical text.

The main responsibilities of Person 2 are:

- Validate the dataset structure
- Check required columns
- Identify missing values
- Check duplicate records
- Check empty text fields
- Generate dataset statistics
- Analyze medical-specialty distribution
- Analyze transcription length
- Provide a validated dataset for downstream NLP processing

---

## Dataset Location

The medical transcription dataset is stored centrally in the NutriCheck project:

```text
data/
└── raw/
    └── medical_transcriptions.csv