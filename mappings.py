market_viability_mapping = {
    0: "Not Viable",  # Not a viable software business opportunity
    1: "Viable"       # Represents a viable software business opportunity
}

# Business model categories (representing software business models)
business_model_mapping = {
    0: "Not Viable",        # No clear software business model
    1: "SaaS",              # Software as a Service opportunity
    2: "Content/Media",     # Content, media, or information products delivered via software
    3: "Marketplace",       # Software platform connecting two sides
    4: "Community",         # Community-driven software business
    5: "API/Integration",   # Technical service or integration
    6: "Info Product"       # Software-delivered courses, guides, educational content
}

# Time to MVP levels for software products
time_to_mvp_mapping = {
    0: "Undefined",         # For posts that aren't assessed or don't fit
    1: "Weekend Project",   # Can be built in 1-2 days
    2: "Week Project",      # Can be built in a week
    3: "Month Project",     # Requires about a month
    4: "Quarter Project"    # Takes 3+ months
}