# Boolean mapping for is_problem (now representing viable opportunities)
is_problem_mapping = {
    0: "Not Viable",  # Not a viable business opportunity
    1: "Viable"       # Represents a viable business opportunity
}

# Problem type categories (now representing business models)
problem_type_mapping = {
    0: "Not Viable",        # No clear business model
    1: "SaaS",             # Software as a Service opportunity
    2: "Content/Media",     # Content, media, or information products
    3: "Marketplace",       # Platform connecting two sides
    4: "Community",         # Community-driven business
    5: "API/Integration",   # Technical service or integration
    6: "Info Product"      # Courses, guides, educational content
}

# Difficulty levels (now representing time to MVP)
difficulty_mapping = {
    0: "Weekend Project",   # Can be built in 1-2 days
    1: "Week Project",      # Can be built in a week
    2: "Month Project",     # Requires about a month
    3: "Quarter Project"    # Takes 3+ months
}

# Detailed descriptions for each difficulty level
difficulty_criteria = {
    0: """
    Weekend Project:
    - Single developer can build
    - Uses existing APIs/tools
    - Minimal custom code
    - Launch-ready in 48 hours
    - Simple monetization
    """,

    1: """
    Week Project:
    - Small feature set
    - Standard tech stack
    - Basic monetization
    - Limited integrations
    - Quick market testing
    """,

    2: """
    Month Project:
    - Full feature set
    - Multiple integrations
    - Custom development
    - Marketing website
    - Payment processing
    - Basic automation
    """,

    3: """
    Quarter Project:
    - Complex features
    - Multiple platforms
    - Advanced automation
    - Multiple payment options
    - Significant marketing needs
    - Complex operations
    """
}