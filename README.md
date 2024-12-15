# Pydantic AI Example: Invoice Processing Agent

This project demonstrates how to build an AI-powered invoice processing agent using Pydantic AI, showcasing type-safe AI interactions with structured inputs and outputs.

## Overview

This example implements an invoice processing system that:

- Extracts total amounts from invoice images
- Uses OpenAI's multimodal LLM capabilities
- Provides structured, type-safe outputs
- Includes comprehensive test coverage

## Key Features

- Type-safe AI interactions using Pydantic models
- Structured dependency injection with dataclasses
- Async support for API operations
- OpenAI GPT-4 Vision integration
- Tool-augmented AI agent capabilities

## Code Structure

The main components include:

- `MultimodalLLMService`: Service for interacting with OpenAI's vision models
- `InvoiceProcessingDependencies`: Dependency container for the AI agent
- `InvoiceExtractionResult`: Structured output model for extracted data
- Custom tools for processing invoice images

## Prerequisites

Before running this project, ensure you have the following prerequisites:

- Python 3.8 or higher
- An OpenAI API key
- Required Python packages (listed in `requirements.txt`)

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Running the Project

To run the project, use the following command:

```bash
python3 app.py
```

## Testing

To run the tests, use the following command:

```bash
pytest
```
