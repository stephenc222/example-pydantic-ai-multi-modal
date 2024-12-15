"""Example of using pydantic_ai with OpenAI's multimodal LLM for invoice processing."""

import os
import base64
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class LineItem(BaseModel):
    """Structured representation of a line item in an invoice."""
    description: str = Field(description="Description of the line item.")
    quantity: int = Field(description="Quantity of the line item.")
    unit_price: float = Field(description="Unit price of the line item.")
    total_price: float = Field(description="Total price for the line item.")


class InvoiceExtractionResult(BaseModel):
    """Structured response for invoice extraction."""
    total_amount: float = Field(description="The total amount extracted from the invoice image.")
    sender: str = Field(description="The sender of the invoice.")
    date: str = Field(description="The date of the invoice.")
    line_items: list[LineItem] = Field(description="The list of line items in the invoice.")


class MultimodalLLMService:
    """Service to interact with OpenAI multimodal LLMs."""

    def __init__(self, model: str):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model

    def get_model_name(self) -> str:
        """Return the name of the model."""
        return self.model

    async def perform_task(self, image_path: str, response_model: type, max_tokens: int = 5000):
        """Send an image and prompt to the LLM and return structured output."""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        messages = [
            {"role": "system", "content": "You are an assistant that extracts details from invoices."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract the details from this invoice."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_model
        )
        return response.choices[0].message.parsed


@dataclass
class InvoiceProcessingDependencies:
    """Dependencies for the invoice processing agent."""
    llm_service: MultimodalLLMService
    invoice_image_path: str


invoice_processing_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=InvoiceProcessingDependencies,
    result_type=InvoiceExtractionResult,
    system_prompt="Extract the total amount, sender, date, and line items from the given invoice image."
)

summary_agent = Agent(
    "openai:gpt-4o-mini",
    result_type=str,
    system_prompt="Summarize the extracted invoice details into a few sentences."
)


@invoice_processing_agent.tool
async def extract_invoice_details(
    ctx: RunContext[InvoiceProcessingDependencies]
) -> InvoiceExtractionResult:
    """Custom tool to extract details from an invoice image."""
    return await ctx.deps.llm_service.perform_task(
        image_path=ctx.deps.invoice_image_path,
        response_model=InvoiceExtractionResult
    )


async def main():
    """Run the invoice processing agent."""
    deps = InvoiceProcessingDependencies(
        llm_service=MultimodalLLMService(model="gpt-4o-mini"),
        invoice_image_path="images/invoice_sample.png"
    )

    result = await invoice_processing_agent.run(
        "Extract the total amount, sender, date, and line items from this invoice.", deps=deps
    )
    print("Structured Result:", result.data)
    print("=" * 100)
    summary = await summary_agent.run(
        "Summarize the invoice details in a few sentences.", message_history=result.new_messages()
    )
    print("Summary:", summary.data)


if __name__ == "__main__":
    asyncio.run(main())
