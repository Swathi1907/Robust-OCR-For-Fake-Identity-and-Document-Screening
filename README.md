# Robust-OCR-For-Fake-Identity-and-Document-Screening

## AI-Based Fake Identity & Document Screening System

This repository supports an SIH 2026 problem statement focused on intelligent border document screening.

## Background
Border checkpoints face common challenges:
- Fake passports and visas
- Altered photographs
- Modified dates of birth
- Tampered visa stamps
- Identity impersonation and multiple identities
- Expired or blacklisted travel documents
- High passenger volume causing delays

Current verification often relies on manual inspection and basic database lookups, which can be slow and error-prone.

## Problem Objective
Build an AI-powered platform that can:
- Analyze identity and travel documents automatically
- Detect signs of tampering or forgery
- Validate extracted data against standards/rules
- Generate a risk score for faster and more accurate decisions

## Expected Solution Modules

### Module 1: OCR Extraction
**Objective:** Extract key information from document images.

**Input documents:**
- Passport
- Visa
- National ID
- Driving license
- Permit documents

**Sample extracted fields:**
- **Passport:** Name, Passport Number, Nationality, Date of Birth, Date of Expiry, Gender
- **Visa:** Visa Number, Visa Type, Entry Validation, Stay Duration

### Module 2: Document Validation
**Objective:** Verify whether extracted information follows official document standards.

### Module 3: Tampering Detection (Core AI Innovation)
**Objective:** Detect digitally or physically altered documents.

**Use cases:**
- Photo replacement
- Text manipulation
- Stamp forgery detection
- Image metadata analysis

### Module 4: Face Verification
**Objective:** Confirm that the document owner matches the presented individual.

## Expected Impact
- Reduce document verification time from minutes to seconds
- Improve detection of forged and tampered documents
- Standardize screening decisions across checkpoints
- Enable data-driven risk assessment
- Create a digital trail for investigation and intelligence analysis
