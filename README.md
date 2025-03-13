# Pseudonymization Fork Details

This fork builds on the work presented in [_De-Identification of Medical Imaging Data: A Comprehensive Tool for Ensuring Patient Privacy_](https://arxiv.org/abs/2410.12402).

Namely, changing the codebase functionality in the following ways:

- **Introduces a new operator—`PSEUDO`—to the DICOM metadata tag anonymization capabilities (profiles).**

  - This operator allows some types of DICOM tags to have their values replaced with pseudonymized (faked) data.
  - This will ensure **no errors** are caused by invalid (blank) header values when the resulting DICOM file is used in real-world medical applications.
  - Multiple pseudonymization **classes** are available for use in different scenarios, such as:
    - `GIVEN_NAME`
    - `SURNAME`
    - `FULL_NAME`
    - `LOCATION_CA`
    - `LOCATION_BC`
    - `DATE`
    - `TIME`
    - `DATETIME`
    - `ID_NUMERICAL`
    - `ID_ALPHANUMERICAL`
    - `UID`
    - `HEALTH_INSTITUTION_CA`
  - Usage syntax: `PSEUDO [tagName] [PSEUDO_CLASS]`
  - Example usage:
    ```
    FORMAT dicom
    %header
    PSEUDO ClinicalTrialSiteID ID_NUMERICAL
    PSEUDO ClinicalTrialSiteName HEALTH_INSTITUTION
    PSEUDO ConsultingPhysicianName FULL_NAME
    PSEUDO StudyDate DATE
    ```
  - Further documentation is available in [`dicom_deid/deid_options/documentation.md`](dicom_deid/deid_options/documentation.md).

- **Add DICOM anonymization profile reviewer, which introduces logic to check if a new profile contradicts the master profile.**
  - Ensures no operations are attempted that do not comply with the DICOM standard.
- **Changes the Optical Character Recognition (OCR) algorithm from _Tesseract_ to the much more modern Visual Document Understanding (VDU) model, [_Donut_](https://arxiv.org/abs/2111.15664).**
  - Will lead to a higher accuracy in detecting (and therefore removing) baked-in text within DICOM images.
  - Provides an understanding of the context for detected text, which _may_ allow the model to only remove _specific_ text that is sensitive.
- **Provide a secure framework for tracing back the anonymized data to the original patient data at the source.**
  - This traceability is _required_ in many Canadian jurisdictions.
  - Process to access the original data should not be implemented in this application. Instead, a separate, standalone tool will be created for compartmentalization, where only authorized users—through the proper legal or clinical pathways—can re-identify anonymized data if necessary.

[The original repository](https://github.com/TIO-IKIM/medical_image_deidentification) details (`README`) are available below.

> Future fork of this fork will disable all features that do not involve files in DICOM format. This will allow the pipeline to better integrate into clinical environments—which solely use DICOM files in practice—and reduce overhead when dealing with thousands of files.

---

# De-Identification of Medical Imaging Data: A Comprehensive Tool for Ensuring Patient Privacy

[![Python 3.11.2](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Open Source Love][0c]
[![Docker](https://img.shields.io/badge/-Docker-46a2f1?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/r/morrempe/hold)

[0c]: https://badges.frapsoft.com/os/v2/open-source.svg?v=103

<div align="center">

[Getting started](#getting-started) • [Usage](#usage) • [Citation](#citation)

</div>

This repository contains the **De-Identification of Medical Imaging Data: A Comprehensive Tool for Ensuring Patient Privacy**, which enables the user to anonymize a wide variety of medical imaging types, including Magnetic Resonance Imaging (MRI), Computer Tomography (CT), Ultrasound (US), Whole Slide Images (WSI) or MRI raw data (twix).

<div align="center">

![Overview](Figures/aam_pipeline-2.png)

</div>

This tool combines multiple anonymization steps, including metadata deidentification, defacing and skull-stripping while being faster than current state-of-the-art deidentification tools.

![Computationtimes](Figures/computation_times.png)

## Getting started

You can install the anonymization tool either directly via git, by cloning this repository or via Docker.

### Installation via Git

1. Clone repository:

   git clone https://github.com/TIO-IKIM/medical_image_deidentification.git

2. Create a conda environment with Python version 3.12.4 and install the necessary dependencies:

   conda create -n my_env python=3.12.4 --file requirements.txt
   In case of installation issues with conda, use pip install -r requirements.txt to install the dependecies.

3. Activate your new environment:

   conda activate my_env

4. Run the script with the corresponding cli parameter, e.g.:

   python3 deidentify.py [your flags]

### Installation via Docker

Alternatively this tool is distributed via docker. You can find the docker images [here](https://hub.docker.com/r/morrempe/hold). The docker image is available for amd64 and arm64 platforms.

For the installation and execution of the docker image, you must have [Docker](https://docs.docker.com/get-docker/) installed on your system.

1. Pull the docker image

   docker pull morrempe/hold:[tag] (either arm64 or amd64)

2. Run the docker container with attached volume. Your data will be mounted in the `data` folder:

   docker run --rm -it -v [Path/to/your/data]:/data morrempe/hold:[tag]

3. Run the script with the corresponding cli parameter, e.g.:

   python3 deidentify.py [your flags]

## Usage

**De-Identification CLI**

```
usage: deidentify.py [-h] [-v | --verbose | --no-verbose] [-t | --text-removal | --no-text-removal] [-i INPUT]
                                    [-o OUTPUT] [--gpu GPU] [-s | --skull_strip | --no-skull_strip] [-de | --deface | --no-deface]
                                    [-tw | --twix | --no-twix] [-p PROCESSES]
                                    [-d {basicProfile,cleanDescOpt,cleanGraphOpt,cleanStructContOpt,rtnDevIdOpt,rtnInstIdOpt,rtnLongFullDatesOpt,rtnLongModifDatesOpt,rtnPatCharsOpt,rtnSafePrivOpt,rtnUIDsOpt} [{basicProfile,cleanDescOpt,cleanGraphOpt,cleanStructContOpt,rtnDevIdOpt,rtnInstIdOpt,rtnLongFullDatesOpt,rtnLongModifDatesOpt,rtnPatCharsOpt,rtnSafePrivOpt,rtnUIDsOpt} ...]]

options:
  -h, --help            show this help message and exit
  -v, --verbose, --no-verbose
  -t, --text-removal, --no-text-removal
  -i INPUT, --input INPUT
                        Path to the input data.
  -o OUTPUT, --output OUTPUT
                        Path to save the output data.
  --gpu GPU             GPU device number. (default 0)
  -s, --skull_strip, --no-skull_strip
  -de, --deface, --no-deface
  -tw, --twix, --no-twix
  -w, --wsi, --no-wsi
  -p PROCESSES, --processes PROCESSES
                        Number of processes to use for multiprocessing.
  -d {basicProfile,cleanDescOpt,cleanGraphOpt,cleanStructContOpt,rtnDevIdOpt,rtnInstIdOpt,rtnLongFullDatesOpt,rtnLongModifDatesOpt,rtnPatCharsOpt,rtnSafePrivOpt,rtnUIDsOpt} [{basicProfile,cleanDescOpt,cleanGraphOpt,cleanStructContOpt,rtnDevIdOpt,rtnInstIdOpt,rtnLongFullDatesOpt,rtnLongModifDatesOpt,rtnPatCharsOpt,rtnSafePrivOpt,rtnUIDsOpt} ...], --deidentification-profile {basicProfile,cleanDescOpt,cleanGraphOpt,cleanStructContOpt,rtnDevIdOpt,rtnInstIdOpt,rtnLongFullDatesOpt,rtnLongModifDatesOpt,rtnPatCharsOpt,rtnSafePrivOpt,rtnUIDsOpt} [{basicProfile,cleanDescOpt,cleanGraphOpt,cleanStructContOpt,rtnDevIdOpt,rtnInstIdOpt,rtnLongFullDatesOpt,rtnLongModifDatesOpt,rtnPatCharsOpt,rtnSafePrivOpt,rtnUIDsOpt} ...]
                        Which DICOM deidentification profile(s) to apply. (default None)
```

## Citation

If you use our tool in your work, please cite us with the following BibTeX entry.

```latex
@misc{rempe2024deidentificationmedicalimagingdata,
      title={De-Identification of Medical Imaging Data: A Comprehensive Tool for Ensuring Patient Privacy},
      author={Moritz Rempe and Lukas Heine and Constantin Seibold and Fabian Hörst and Jens Kleesiek},
      year={2024},
      eprint={2410.12402},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2410.12402},
}
```
