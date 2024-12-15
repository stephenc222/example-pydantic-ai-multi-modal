"""Test the invoice processing agent and summary tool with mock LLM services."""

import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel
from app import (
    InvoiceExtractionResult,
    LineItem,
    invoice_processing_agent,
    summary_agent,
    InvoiceProcessingDependencies
)

pytestmark = pytest.mark.asyncio  # Mark all tests as async

models.ALLOW_MODEL_REQUESTS = False


class MockMultimodalLLMService:
    """Mock LLM service that returns a fixed response for testing."""
    
    async def perform_task(self, image_path: str, response_model: type, max_tokens: int = 100):
        """Return a mock response with predefined data."""
        return response_model(
            total_amount=123.45,
            sender="Test Sender",
            date="2023-10-01",
            line_items=[
                LineItem(description="Item 1", quantity=1, unit_price=100.0, total_price=100.0),
                LineItem(description="Item 2", quantity=2, unit_price=11.725, total_price=23.45)
            ]
        )


class MockFailingLLMService:
    """Mock LLM service that simulates file not found errors."""
    
    async def perform_task(self, image_path: str, response_model: type, max_tokens: int = 100):
        """Simulate a file not found error."""
        raise FileNotFoundError("Image file not found")


class ZeroAmountLLMService:
    """Mock LLM service that returns zero amount for testing edge cases."""
    
    async def perform_task(self, image_path: str, response_model: type, max_tokens: int = 100):
        """Return a mock response with zero total amount."""
        return response_model(
            total_amount=0.0,
            sender="Test Sender",
            date="2023-10-01",
            line_items=[]
        )


async def test_invoice_extraction():
    """Test the invoice processing agent with a mock LLM service."""
    deps = InvoiceProcessingDependencies(
        llm_service=MockMultimodalLLMService(),
        invoice_image_path="invoice_sample.png",
    )

    with invoice_processing_agent.override(
        model=TestModel(custom_result_args={
            "total_amount": 123.45,
            "sender": "Test Sender",
            "date": "2023-10-01",
            "line_items": [
                LineItem(description="Item 1", quantity=1, unit_price=100.0, total_price=100.0),
                LineItem(description="Item 2", quantity=2, unit_price=11.725, total_price=23.45)
            ]
        })
    ):
        result = await invoice_processing_agent.run(
            "Extract the total amount, sender, date, and line items from this invoice.",
            deps=deps
        )

    assert isinstance(result.data, InvoiceExtractionResult)
    assert result.data.total_amount == 123.45
    assert result.data.sender == "Test Sender"
    assert result.data.date == "2023-10-01"
    assert len(result.data.line_items) == 2
    assert result.data.line_items[0].description == "Item 1"
    assert result.data.line_items[1].description == "Item 2"


async def test_invoice_extraction_invalid_image():
    """Test invoice processing with invalid image path."""
    deps = InvoiceProcessingDependencies(
        llm_service=MockFailingLLMService(),
        invoice_image_path="nonexistent.png",
    )

    with pytest.raises(FileNotFoundError):
        with invoice_processing_agent.override(model=TestModel()):
            await invoice_processing_agent.run(
                "Extract the total amount, sender, date, and line items from this invoice.",
                deps=deps
            )


async def test_invoice_extraction_zero_amount():
    """Test invoice processing with zero amount."""
    deps = InvoiceProcessingDependencies(
        llm_service=ZeroAmountLLMService(),
        invoice_image_path="test_invoice.png",
    )

    with invoice_processing_agent.override(model=TestModel(
        custom_result_args={
            "total_amount": 0.0,
            "sender": "Test Sender",
            "date": "2023-10-01",
            "line_items": []
        }
    )):
        result = await invoice_processing_agent.run(
            "Extract the total amount, sender, date, and line items from this invoice.",
            deps=deps
        )

    assert result.data.total_amount == 0.0
    assert result.data.sender == "Test Sender"
    assert result.data.date == "2023-10-01"
    assert len(result.data.line_items) == 0


async def test_summary_tool():
    """Test the summary tool with a mock LLM service."""
    deps = InvoiceProcessingDependencies(
        llm_service=MockMultimodalLLMService(),
        invoice_image_path="invoice_sample.png",
    )

    with invoice_processing_agent.override(
        model=TestModel(custom_result_args={
            "total_amount": 123.45,
            "sender": "Test Sender",
            "date": "2023-10-01",
            "line_items": [
                LineItem(description="Item 1", quantity=1, unit_price=100.0, total_price=100.0),
                LineItem(description="Item 2", quantity=2, unit_price=11.725, total_price=23.45)
            ]
        })
    ):
        extraction_result = await invoice_processing_agent.run(
            "Extract the total amount, sender, date, and line items from this invoice.",
            deps=deps
        )
    with summary_agent.override(model=TestModel()):
        print(extraction_result.new_messages())
        summary_result = await summary_agent.run(
            "Summarize the invoice details in a few sentences.",
            message_history=extraction_result.new_messages()
        )

    assert isinstance(summary_result.data, str)
    assert "123.45" in summary_result.data
    assert "Test Sender" in summary_result.data
    assert "2023-10-01" in summary_result.data
