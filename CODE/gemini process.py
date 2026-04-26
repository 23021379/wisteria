import google.generativeai as genai
from PIL import Image
import os
import json
import time # Import time for potential delays/retries
import statistics
# --- Configuration ---
# Configure the API key (replace with your actual key or use environment variables)
# Make sure to keep your API key secure!
try:
    # Attempt to get API key from environment variable is safer
    # Replace with your actual key if not using environment variables
    api_key = "[REDACTED_BY_SCRIPT]" # IMPORTANT: Keep this secure!
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")
    exit() # Or handle error appropriately

# Select the model
# Gemini Flash is fast but might struggle more with complexity than Pro.
# Consider '[REDACTED_BY_SCRIPT]' if Flash encounters issues.
model = genai.GenerativeModel('[REDACTED_BY_SCRIPT]') # Use the latest flash model

# --- Define Inputs ---
floorplan_image_path = r"[REDACTED_BY_SCRIPT]" # Replace with actual path
#c:\Users\dell\Desktop\property images\190424_32795187_FLP_00_0000.jpeg
image_dir = r"[REDACTED_BY_SCRIPT]" # Directory containing room images

# --- Function to find room images ---
def get_room_images(image_dir, floorplan_path):
    room_image_paths = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    try:
        floorplan_path_norm = os.path.normpath(floorplan_path)
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            file_path_norm = os.path.normpath(file_path)
            if (os.path.isfile(file_path) and
                any(filename.lower().endswith(ext) for ext in image_extensions)):
                if file_path_norm != floorplan_path_norm:
                    room_image_paths.append(file_path)
        print(f"[REDACTED_BY_SCRIPT]'{image_dir}'[REDACTED_BY_SCRIPT]")
        return room_image_paths
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]'{image_dir}'")
        return []
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return []

# --- Load Images ---
def load_image(path):
    try:
        # Attempt to open and verify the image
        img = Image.open(path)
        img.verify() # Verify helps catch some truncated images
        # Reopen after verify
        img = Image.open(path)
        print(f"[REDACTED_BY_SCRIPT]")
        return img
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
        return None
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return None

floorplan_image = load_image(floorplan_image_path)
room_image_paths = get_room_images(image_dir, floorplan_image_path)
room_images = [load_image(p) for p in room_image_paths]
# Filter out None values if any images failed to load
room_images = [img for img in room_images if img is not None]

if not floorplan_image:
    print("[REDACTED_BY_SCRIPT]")
    exit()
if not room_images:
    print("[REDACTED_BY_SCRIPT]")
    exit()
# Create indexed list for prompts - IMPORTANT for prompt consistency
# Ensures "Image 1" in the prompt corresponds to the first image in room_images etc.
indexed_room_images = {f"Image {i+1}": img for i, img in enumerate(room_images)}
print(f"[REDACTED_BY_SCRIPT]")



persona_details_string = """
* Persona 1: The First-Time Buyer (Solo) Persona Identity & Demographics: Name: Liam O’Connell Age: 27 Occupation: Junior Marketing Executive Family Situation: Single, no dependents. General Location/Living Context: Currently renting a room in a shared house in Bristol. Looking to buy his first property (likely a 1- or 2-bedroom flat or small terraced house) within a 30-40 minute commute (cycling or public transport) of Bristol city centre. Income Level/Budget: Earns £32,000 per year. Has saved a deposit with help from family and potentially using a Lifetime ISA (LISA). Maximum budget is £220,000. Psychographics & Lifestyle: Values independence and achieving the milestone of homeownership. Enjoys socialising with friends, exploring the city'[REDACTED_BY_SCRIPT]"wasting money on rent,"[REDACTED_BY_SCRIPT]'t need to be perfect, but wants good "bones" for cosmetic updates (e.g., painting, new flooring) without major structural work, Low Maintenance (Structural): Wants to avoid immediate large expenses like a new roof or boiler. A good Energy Performance Certificate (EPC) rating (C or above) is desirable for lower bills and future-proofing. Pain Points: Hidden Costs: Worried about unexpected repair bills soon after moving in, Lack of Space: Hates feeling cramped, needs efficient use of space, especially storage. *
* Persona 2: The Young Family (Parents with Infants/Toddlers) Persona Identity & Demographics: Names: Aisha Khan (36, GP returning from maternity leave part-time), Ben Carter (38, Software Developer). Children: Zara (2 years), Noah (6 months). Family Situation: Married couple with two young children. General Location/Living Context: Currently in a 2-bed garden flat in a busy area of Birmingham. Seeking a 3- or 4-bedroom house with a garden in a quieter, family-friendly suburb (e.g., Solihull, Sutton Coldfield, or surrounding villages) with good local amenities and schools. Income Level/Budget: Combined household income of £110,000 per year. Budget £500,000 - £550,000. Psychographics & Lifestyle: Family life is the central focus. Values safety, community, and providing a nurturing environment for their children. Lifestyle is hectic, juggling childcare, work (Ben is hybrid, Aisha mainly practice-based), and household tasks. Seek practicality and convenience. Increasingly aware of sustainability for their children'[REDACTED_BY_SCRIPT]'s sleep, Lack of Storage: Current flat feels constantly cluttered with essential baby/toddler items. *
* Persona 3: The Downsizers (Empty Nesters) Persona Identity & Demographics: Names: Susan Jenkins (65, Recently Retired Librarian), Graham Jenkins (67, Retired Engineer). Children: Two adult children living independently elsewhere in the country. Family Situation: Married couple, empty nesters. General Location/Living Context: Selling their long-term 4-bedroom detached family home with a large garden in Cheshire. Seeking a smaller, manageable 2-bedroom property (bungalow, ground floor flat, or modern apartment with lift) in or near a vibrant market town (e.g., Shrewsbury, Ludlow) with good amenities and transport links. Income Level/Budget: Combined income from state and private pensions, plus savings, approx £45,000 per year. Expecting to release significant equity from their house sale, setting a purchase budget of £380,000 (allowing funds for travel/hobbies). Psychographics & Lifestyle: Value comfort, security, independence, and ease of living. Look forward to enjoying retirement with less responsibility for property upkeep. Interests include travel, reading (Susan), gentle hiking/local history (Graham), and occasional visits from grandchildren. Seek a friendly community feel and convenience. Not overly concerned with cutting-edge design but appreciate modern comforts and efficiency. Goals & Context for Property: Downsizing primarily to reduce the burden of maintaining a large house and garden, release equity for retirement lifestyle, and live somewhere convenient for amenities and potentially future care needs. Considering future-proofing for mobility. Needs, Priorities & Pain Points: Needs/Priorities: Low Maintenance: Minimal garden upkeep (patio or small lawn fine); modern construction preferred; recently updated kitchen/bathroom desirable, Accessibility: Single-level living strongly preferred (bungalow); if an apartment, reliable lift access is essential; walk-in shower favoured over bath, Proximity to Amenities: Easy walking distance (<15 mins) or short bus ride to shops, GP surgery, library, cafes, and potentially a train station for travel, Comfort & Efficiency: Good insulation, efficient and reliable heating system (e.g., modern combi boiler), double glazing. EPC rating B desirable. Pain Points: Stairs: Increasingly seen as a daily inconvenience and potential future hazard, Excessive Chores: Want to spend time on leisure, not demanding DIY or gardening tasks. *
* Persona 4: The Remote Worker Persona Identity & Demographics: Name: Morgan Ellis (uses they/them pronouns) Age: 42 Occupation: Senior Policy Advisor (Fully Remote for a London-based charity) Family Situation: Single, no children. Has a close network of friends spread across the country. General Location/Living Context: Currently renting a modern 2-bed flat in Manchester city centre. Seeking to buy a 2- or 3-bedroom property with character, potentially semi-rural or edge-of-town, anywhere in Northern England or Scotland that offers excellent broadband and access to nature (e.g., Peak District fringes, Yorkshire Dales, Scottish Borders). Less constrained by traditional commuter concerns. Income Level/Budget: Earns £65,000 per year. Budget £400,000. Psychographics & Lifestyle: Values autonomy, work-life balance, intellectual stimulation, and environmental sustainability. Highly organised and self-disciplined regarding work. Passionate about hillwalking, wild swimming, and conservation. Seeks peace and quiet for focused work but also wants easy access to outdoor pursuits. Tech-literate and reliant on digital connectivity. Appreciates character and quality over minimalist modernism. Goals & Context for Property: Buying a property that fully supports a permanent remote working lifestyle, offering a dedicated, comfortable, and inspiring home office space. Wants a home that facilitates their hobbies and provides a better connection to the natural environment than their current city-centre flat. Location is secondary to the property's suitability for work and lifestyle needs. Needs, Priorities & Pain Points: Needs/Priorities: Dedicated, Quiet Home Office: A separate room (not a corner of the living room) with good natural light, sufficient space for desk/storage, and crucially, quiet enough for video calls and focused work, Reliable High-Speed Internet: Non-negotiable. Must verify availability of Fibre-to-the-Premises (FTTP) or minimum reliable superfast broadband (>50Mbps download). Will check Ofcom checker and provider availability specifically, Access to Nature: Direct access to, or very close proximity (<10 min drive) to, walking trails, hills, forests, or coastline. A decent-sized garden is also a plus, Work-Life Separation: A layout that allows distinct separation between the workspace and relaxation/living areas. Pain Points: Poor/Unreliable Internet Connectivity: Biggest potential deal-breaker for their work, Compromised Workspace: Trying to work effectively in a shared-use space or a dark, cramped room. *
* Persona 5: The First-Time Buyer Couple Persona Identity & Demographics: Names: Maya Sharma & Ben Carter Age Range: Maya (29), Ben (30) Occupation/Status: Maya (Junior Marketing Manager), Ben (Electrician, self-employed) Family Situation: Committed couple, living together for 3 years, no children yet, planning to get a dog within a year, potentially children in 3-5 years. Location/Context: Currently renting a 1-bedroom flat in a city fringe area (e.g., Zone 3 London, or equivalent major city suburb). Looking to buy their first property in a similar or slightly further out area offering better value. Income/Budget: Combined gross income ~£75,000. Maximum budget ~£350,000 (reliant on high LTV mortgage). Budget is a major constraint. Psychographics & Lifestyle: Values: Financial stability, creating a home base, practicality, future planning, value-for-money. Ben values reliability and low-maintenance; Maya values aesthetics and potential for personalisation. Attitudes: Eager and excited but also anxious about the financial commitment and complexity of buying. Cautious investors. Ben is pragmatic; Maya is more aspirational but grounded by the budget. Interests/Lifestyle: Busy work weeks. Weekends involve DIY projects (small scale currently), exploring local parks/cafes, socialising with friends (often hosting casually), planning future travel. Ben enjoys football; Maya enjoys yoga and social media (design/home accounts). Conscious of monthly outgoings. Goals & Context for Property: Reason: Buying their first home to escape renting, build equity, and gain more space. Primary Goal: Find a 2-bedroom property (house or flat with own entrance/garden ideally) that they can afford, is structurally sound, offers scope for cosmetic updates over time, and is in a safe neighbourhood with reasonable transport links for their commutes. See it as a 5-7 year home before potentially needing more space. Needs, Priorities & Pain Points: Needs/Priorities: Budget Adherence: Non-negotiable. Must fit within their max borrowing, Second Bedroom: Essential for WFH space (Maya sometimes works remotely) / future nursery / guest room, Potential for Improvement: Scope for cosmetic updates they can do themselves to add value/personalise, Decent Kitchen/Bathroom Condition: Functional and clean, even if not brand new, to avoid immediate large expenses, Some Outdoor Space: Small garden, patio, or balcony (for future dog/sitting out). Pain Points: Hidden Costs: Fearful of unexpected major repairs (e.g., boiler, roof), Lack of Storage: Current rental is cramped; they hate clutter. *
* Persona 6: The Education-Conscious Parents Persona Identity & Demographics: Names: Dr. Aisha Khan & Mr. David Miller Age Range: Aisha (42), David (44) Occupation/Status: Aisha (General Practitioner), David (University Lecturer) Family Situation: Married with two children aged 8 and 10. Location/Context: Currently own a 3-bed semi-detached house in a good, but not top-tier, school catchment area. Looking to move specifically for secondary school access. Income/Budget: Combined gross income ~£150,000. Budget for new home up to £750,000 (significant equity from current home + comfortable mortgage). Psychographics & Lifestyle: Values: Education, family well-being, stability, community involvement, intellectual pursuits, quality time. Attitudes: Diligent researchers (especially regarding schools/neighbourhoods), willing to stretch budget for the 'right' location, risk-averse regarding safety and long-term investment value. Prioritise substance over flashy style. Interests/Lifestyle: Children'[REDACTED_BY_SCRIPT]'s homework and parents' work, Safe Neighbourhood: Low crime rates, family-friendly amenities nearby (parks, library), Garden Space: Secure area for children to play. Pain Points: Busy Roads / Traffic Noise: Concerns about child safety and disturbance, Lack of Functional Layout: Dislike disjointed homes where family members feel too separated. *
* Persona 7: The Active Retiree Persona Identity & Demographics: Name: Eleanor Vance Age Range: 68 Occupation/Status: Retired Head Teacher. Widowed 5 years ago. Financially comfortable (pension + invested savings + equity from sold family home). Family Situation: Two adult children living elsewhere, three young grandchildren visit occasionally. Lives alone but socially active. Location/Context: Sold large family home in the suburbs. Looking to downsize to a smaller, more manageable property in a vibrant town or village with good amenities. Income/Budget: Effectively budget-driven by desired lifestyle rather than income constraint. Looking at properties around £450,000 - £550,000, wants low running costs. Psychographics & Lifestyle: Values: Independence, security, community connection, ease of living, staying active (mentally and physically), comfort. Attitudes: Pragmatic, forward-thinking (planning for future mobility needs), wants quality and reliability, appreciates established neighbourhoods, values peace but not isolation. Interests/Lifestyle: Member of local U3A (University of the Third Age), book club, enjoys gardening (manageable scale), walks, theatre trips, occasional travel (national and international), hosts grandchildren periodically. Uses technology confidently (online shopping, video calls). Goals & Context for Property: Reason: Downsizing from a large, high-maintenance family home post-retirement and widowhood. Primary Goal: Find a comfortable, secure, and easy-to-maintain home (2-bedroom bungalow, house, or ground/first-floor apartment with lift) in a location with good amenities (shops, doctor, transport, social clubs) within walking distance. Needs space for hobbies and occasional guests. Needs, Priorities & Pain Points: Needs/Priorities: Ease of Access/Single-Level Living: Minimal stairs essential, ideally bungalow or apartment with lift. Wide doorways/hallways a plus, Low Maintenance: Both property (modern construction, good condition) and garden (small patio/manageable beds), Location & Amenities: Walking distance to shops, GP, bus stop, community centre/social hubs, Security: Safe neighbourhood, secure locks/windows, potentially part of a managed development, Guest Space: Second bedroom for visiting family/friends or hobby use. Pain Points: Steep Stairs / Awkward Layouts: Immediate disqualifier due to future-proofing concerns, Isolation: Locations requiring driving for basic errands or social interaction. *
* Persona 8: The Eco-Conscious Buyer Persona Identity & Demographics: Name: Alex Chen Age Range: 38 Occupation/Status: Sustainability Consultant Family Situation: Single, no children. Has a rescue cat. Location/Context: Currently renting an apartment in a city known for environmental initiatives (e.g., Bristol, Freiburg). Looking to buy a property that aligns with their values. Income/Budget: Gross income ~£65,000. Budget up to £400,000. Willing to invest more upfront for long-term savings and lower impact. Psychographics & Lifestyle: Values: Environmental sustainability (core value), ethical consumption, health & well-being, community resilience, simplicity, resource conservation. Attitudes: Highly informed about green building practices, critical of wastefulness/inefficiency, proactive in seeking sustainable solutions, optimistic about technological solutions but prefers natural/passive methods. Believes individual actions matter. Interests/Lifestyle: Cycles or uses public transport primarily. Committed vegetarian/vegan. Supports local/organic food producers, involved in local environmental groups, enjoys hiking/nature, practices mindfulness/yoga. Minimises waste (recycling, composting). Reads environmental news/blogs avidly. Goals & Context for Property: Reason: Buying a property to align living situation with deeply held environmental values, seeking long-term stability and lower running costs. Primary Goal: Find a property (house or eco-conscious apartment) with strong sustainability credentials (existing or potential). Prioritises energy efficiency, use of natural/sustainable materials, and features supporting a low-impact lifestyle (e.g., garden space, bike storage, good public transport). Needs, Priorities & Pain Points: Needs/Priorities: Energy Efficiency: High EPC rating (B or ideally A), good insulation, quality double/triple glazing, potential for renewables (solar PV/thermal), Sustainable Materials: Evidence of natural/recycled/low-VOC materials where possible (wood floors, natural paints etc.), Proximity to Public Transport/Cycle Paths: Essential for low-car lifestyle, Potential for Gardening: Space for vegetable patch/composting, even if small, Good Natural Light/Ventilation: Reduces need for artificial lighting/heating/cooling. Pain Points: Poor Energy Performance (Low EPC): Major deterrent due to cost and environmental impact, Reliance on Fossil Fuels: E.g., old oil heating systems, gas hobs (prefers induction). *
* Persona 9: The Fixer-Upper Fanatic Persona Identity & Demographics: Names: Sam Davies & Jo Fletcher Age Range: Sam (33), Jo (35) Occupation/Status: Sam (Carpenter/Joiner), Jo (Project Manager - Construction Industry) Family Situation: Couple, potentially planning family in future but property comes first. Have a dog. Location/Context: Currently renting cheaply or living with family to save deposit. Looking for a property with significant renovation potential in an up-and-coming or established area where fully renovated properties would be out of budget. Income/Budget: Combined gross income ~£90,000. Property budget up to £300,000, plus a separate renovation budget of ~£50k-£70k (part savings, part potentially further borrowing/phased work). Psychographics & Lifestyle: Values: Hard work, tangible results, creativity, potential, adding value, self-sufficiency. Not afraid of disruption. Attitudes: See potential where others see problems. Enjoy the process of transformation. Confident in their skills (Sam) and management abilities (Jo). Resourceful and budget-savvy. Realistic about the challenges of renovation. Interests/Lifestyle: Weekends and evenings are often dedicated to DIY/planning. Follow renovation blogs/TV shows. Enjoy browsing reclamation yards/architectural salvage. Practical and hands-on. Social life might take a back seat during intensive project phases. Prioritise functionality and good 'bones' over superficial finishes. Goals & Context for Property: Reason: To buy a property significantly below market value for its condition, allowing them to renovate it to their own taste and build substantial equity. Primary Goal: Find a structurally sound property (house, likely 2-3 bed) needing extensive cosmetic or even some structural renovation. Focus is on location, basic layout potential, and 'good bones' (roof, walls, foundations). Needs, Priorities & Pain Points: Needs/Priorities: Structural Integrity: Sound foundations, roof, main walls are paramount, Renovation Potential: Obvious scope for improvement (outdated kitchen/bathroom, poor layout, bad decor), Location: Good neighbourhood or area with potential for price growth, Good 'Bones'/Layout Potential: Basic room sizes and arrangement offer scope for reconfiguration without major structural changes (e.g., knocking down non-load bearing walls is okay), Space for Work/Storage: During renovation, need space to work and store materials. A garage or large shed is a bonus. Pain Points: Irremediable Structural Issues: Problems too costly or complex to fix (e.g., severe subsidence, major damp requiring tanking), Poorly Done Previous Renovations: Bodged DIY that needs ripping out and re-doing correctly is frustrating and costly. *
* Persona 10: The Established Family (Parents with Teenagers) Persona Identity & Demographics: Names: David (52) and Eleanor (50) Miller Occupation: David (Civil Engineer), Eleanor (Secondary School Teacher) Family Situation: Two teenagers, Sophie (17, studying for A-levels) and Tom (15, GCSE years). Location/Context: Currently in a 3-bedroom semi-detached house in a suburb of Bristol, UK. Finding it increasingly cramped as children need study space and social areas. Income/Budget: Combined income approx. £110,000. Budget up to £750,000 for the new home. Psychographics & Lifestyle: Values education highly (driven by Eleanor'[REDACTED_BY_SCRIPT]'s A-levels). Prioritize family connection but recognize teens'[REDACTED_BY_SCRIPT]"on top of each other"; no personal space, Teenagers taking over the main living room. *
* Persona 11: The Upsizers Persona Identity & Demographics: Names: Chloe (38) and Ben (40) Davies Occupation: Chloe (Marketing Manager), Ben (Owns a small graphic design business, works from home). Family Situation: One child (Leo, 6) and expecting a second. Currently own a 2-bedroom terraced house. Location/Context: Living in a trendy but compact neighbourhood (e.g., Southville, Bristol). Seeking a larger home in a more family-oriented suburb (e.g., Bishopston/Westbury-on-Trym, Bristol) with better primary schools and more green space. Income/Budget: Combined income approx. £130,000. Budget up to £800,000, leveraging equity from current home. Psychographics & Lifestyle: Value family life, community, and space for children to grow. Appreciate quality finishes and design (influenced by Ben'[REDACTED_BY_SCRIPT]'s work disrupted by lack of dedicated office, Outdated fixtures/finishes in current home. *
* Persona 12: The Urban Professional (Couple) Persona Identity & Demographics: Names: Aisha Khan (34) and Liam Smith (35) Occupation: Aisha (Lawyer), Liam (Software Developer at a FinTech company). Family Situation: Couple, no children (DINK - Dual Income No Kids). Location/Context: Currently renting a modern apartment in a central city location (e.g., Spinningfields, Manchester). Looking to buy their first property together in a similar central or very well-connected urban area. Income/Budget: Combined income approx. £150,000. Budget up to £550,000. Psychographics & Lifestyle: Value convenience, proximity to work, restaurants, bars, culture. Enjoy a busy social life. Appreciate modern, high-spec design and technology (smart home features are a plus). Career-focused and time-poor, therefore low-maintenance living is essential. Travel frequently for work/leisure. Lifestyle involves long work hours, dining out, socializing, gym, occasional weekends away. Home is a base and a place to relax/entertain close friends, not a family hub. Goals & Context for Property: Reason: Transitioning from renting to owning, seeking a property that fits their urban lifestyle and serves as a good long-term investment. Primary Goal: To buy a high-specification 2-bedroom apartment or modern townhouse in a prime central location or an area with excellent transport links, featuring contemporary design, quality finishes, and potentially amenities like a gym or concierge. Needs, Priorities & Pain Points: Needs/Priorities: Prime location (walkable to work/transport/amenities), Modern, high-quality finishes (kitchen, bathroom), Good layout for occasional entertaining (open plan living), Low maintenance (e.g., no large garden, service charges acceptable), Good natural light / interesting views (desirable). Pain Points: Wasting money on rent, Poor quality finishes in previous rentals, Long commutes from less central locations. *
* Persona 13: The Luxury Seeker Persona Identity & Demographics: Names: Julian Vance (58) Occupation: Founder/CEO of a successful tech company (post-exit or semi-retired). Family Situation: Divorced, two adult children who visit occasionally. Location/Context: Selling a large family home in the home counties. Seeking a statement penthouse apartment in a prime central London location (e.g., Mayfair, Knightsbridge) or a unique architect-designed house in an exclusive area (e.g., Sandbanks, Poole). Income/Budget: High Net Worth Individual (HNWI). Budget £5M - £10M+. Psychographics & Lifestyle: Values exclusivity, privacy, security, cutting-edge design, and high-quality materials. Appreciates unique architectural features and bespoke finishes. Views property as a status symbol and part of an investment portfolio. Enjoys fine dining, art collection, travel, and requires space suitable for high-profile entertaining. Wants "turn-key" perfection and premium concierge/services. Lifestyle involves international travel, board meetings, high-end social events, philanthropy. Home needs to be both a private sanctuary and an impressive entertaining venue. Goals & Context for Property: Reason: Seeking a primary residence that reflects personal success and caters to a specific luxury lifestyle, combining privacy with impressive entertaining capabilities. Also considered a significant asset within a wealth portfolio. Primary Goal: To acquire an exceptional property (penthouse or unique house) in a globally recognized prime location, offering bespoke design, state-of-the-art technology/security, premium amenities (e.g., pool, views, concierge), absolute privacy, and significant prestige value. Needs, Priorities & Pain Points: Needs/Priorities: Prime, prestigious location, Exceptional design & unique architectural features, High-end, bespoke finishes and materials, State-of-the-art security and privacy features, Premium amenities (concierge, private gym/pool, stunning views). Pain Points: Lack of privacy or security, Standard/mass-market finishes or design, Compromise on location prestige. *
* Persona 14: The Investor (Buy-to-Let) Persona Identity & Demographics: Name: Michael Chen (42) Occupation: Portfolio Landlord / Small Business Owner (operates alongside property investments). Family Situation: Married, two young children. Location/Context: Lives in Southeast England but invests in properties across the UK, focusing on areas with strong rental demand and yield potential (e.g., Northern cities, Midlands university towns). Income/Budget: Uses a combination of personal funds, business profits, and buy-to-let mortgages. Focuses on properties typically under £250,000 where yields are stronger. Psychographics & Lifestyle: Highly analytical and numbers-driven. Values ROI (Return on Investment), rental yield, and capital growth potential. Risk-averse regarding void periods and maintenance costs. Focuses on durability and broad tenant appeal rather than personal taste. Stays informed about market trends, regulations, and tax implications. Lifestyle involves researching investment opportunities, managing existing properties (often via agents), dealing with finances. Property investment is a core part of their financial strategy. Goals & Context for Property: Reason: To acquire properties that generate consistent rental income (positive cash flow after costs) and offer potential for long-term capital appreciation. Primary Goal: To purchase a property (typically 2-3 bed house or flat) in an area with proven high tenant demand (e.g., near transport hubs, universities, major employers), offering a strong gross rental yield (aiming for 6%+), requiring minimal immediate renovation, and featuring durable, low-maintenance finishes. Needs, Priorities & Pain Points: Needs/Priorities: Strong Rental Yield Potential, High Tenant Demand / Location relevant to renters, Durability of fixtures and fittings, Low ongoing maintenance requirements, Potential for capital growth (secondary to yield). Pain Points: Rental void periods (property empty), High or unexpected maintenance costs eating into profit, Properties that are difficult to rent due to niche appeal or poor location. *
* Persona 15: The Multi-Generational Household Persona Identity & Demographics: Names: The Sharma Family - Grandparents: Anil (75, retired) & Priya (72, retired, minor mobility issues); Parents: Rohan (48, Accountant) & Meera (46, Pharmacist); Child: Maya (12). Occupation: Mixed (active professionals & retirees). Family Situation: Three generations planning to live together. Selling two separate homes to pool resources. Location/Context: Moving from separate homes in the same town (e.g., Leicester) to find one larger property suitable for all. Income/Budget: Combined resources from house sales and incomes. Budget around £900,000. Psychographics & Lifestyle: Value family support, shared responsibilities (childcare, elder support), and cultural connection. However, also value individual privacy and autonomy within the shared home. Need spaces that can adapt to different needs (quiet retirement, busy work/school life, social family gatherings). Aware of potential for friction if spaces aren'[REDACTED_BY_SCRIPT]' needs, Difficulties for Priya navigating stairs or inaccessible bathrooms. *
* Persona 16: The Pet Owner Persona Identity & Demographics: Name: Jessica Miller (31) Occupation: Veterinary Nurse Family Situation: Single, owns a medium-sized active dog (e.g., Springer Spaniel) named Max. Location/Context: Renting a flat with limited outdoor access. Seeking to buy her first home, with the dog'[REDACTED_BY_SCRIPT]'s needs in all plans. Works shifts, so easy access to walks is important. Goals & Context for Property: Reason: To provide a better living environment for herself and her dog, specifically needing secure outdoor space and easier access to walks than her current flat allows. Primary Goal: To buy a 2-3 bedroom house or ground-floor flat with a private, securely fenced garden/patio, durable flooring (e.g., laminate, tile, not easily damaged carpet), located near parks or safe walking routes. Needs, Priorities & Pain Points: Needs/Priorities: Securely enclosed private outdoor space (garden/patio), Durable, easy-to-clean flooring in main living areas/hallway, Proximity to parks, fields, or designated walking trails, Layout allowing easy access to garden (e.g., back door from living area/kitchen), Potential space for dog bed/crate without obstructing walkways. Pain Points: No private outdoor space for the dog, Impractical flooring (e.g., light-coloured carpets) easily ruined by pets, Having to travel far for daily dog walks. *
* Persona 17: The Social Entertainer Persona Identity & Demographics: Names: Mark Jenkins (45) and Antonio Rossi (42) Occupation: Mark (Sales Director), Antonio (Restaurateur) Family Situation: Couple, no children. Location/Context: Own a stylish apartment but find it lacks space for larger gatherings. Looking for a house with dedicated entertaining areas in a lively suburban area with good transport links (e.g., Chorlton, Manchester). Income/Budget: Combined income approx. £180,000. Budget up to £950,000. Psychographics & Lifestyle: Highly social, enjoy hosting dinner parties, BBQs, cocktail evenings frequently. Value spacious, flowing layouts that allow guests to mingle easily. Appreciate a well-equipped kitchen that'[REDACTED_BY_SCRIPT]'atmosphere'[REDACTED_BY_SCRIPT]'s influence). Home is a key part of their social identity. Goals & Context for Property: Reason: Current home is too restrictive for the scale and frequency of entertaining they enjoy. Seeking a property specifically designed or adaptable for hosting larger groups comfortably. Primary Goal: To purchase a house with a large, open-plan kitchen/dining/living area, a well-designed kitchen suitable for serious cooking/hosting, and a functional outdoor entertaining space (patio/deck/garden). Potential for a guest bedroom/suite is a bonus. Needs, Priorities & Pain Points: Needs/Priorities: Large, open-plan main living/dining/kitchen area, High-quality, well-equipped kitchen with good layout (e.g., island), Functional outdoor entertaining space (patio/deck), Good flow between indoor and outdoor spaces, Downstairs WC / Guest-accessible bathroom. Pain Points: Kitchen separated from guests / Host isolated while cooking, Lack of space for guests to mingle comfortably, No suitable outdoor area for summer parties/BBQs. *
* Persona 18: The Minimalist Persona Identity & Demographics: Name: Kenji Tanaka (38) Occupation: UX Designer Family Situation: Single. Location/Context: Moving from a shared flat. Seeking to buy a small but perfectly formed apartment or small house, likely in a city fringe area known for modern design or warehouse conversions (e.g., Shoreditch/Hoxton in London, or Northern Quarter in Manchester). Income/Budget: Income approx. £75,000. Budget up to £450,000. Psychographics & Lifestyle: Values simplicity, order, functionality, and quality over quantity. Believes "less is more". Appreciates clean lines, neutral palettes, natural light, and clever storage solutions. Dislikes clutter intensely. Interested in sustainable design and durable, high-quality materials. Lifestyle is intentional and uncluttered. Prefers experiences over possessions. Works partly from home, needs a calm, organised workspace. Spends time tidying/organising. Goals & Context for Property: Reason: To own a personal space that fully reflects minimalist principles, providing a calm, functional, and aesthetically pleasing environment free from clutter. Primary Goal: To purchase a property (likely 1-2 bedrooms) characterized by clean architectural lines, excellent natural light, a simple, functional layout, and ample, well-designed built-in storage. Quality of finish is more important than size. Needs, Priorities & Pain Points: Needs/Priorities: Clean lines and simple architectural form, Excellent built-in storage solutions, Abundant natural light, High-quality, simple finishes (e.g., wood floors, plain walls), Functional, uncluttered layout. Pain Points: Clutter / Lack of storage, Ornate, fussy, or dated architectural details/finishes, Dark or poorly lit rooms. *
* Persona 19: The Creative/Artist Persona Identity & Demographics: Name: Freya Bellweather (41) Occupation: Freelance Illustrator & Painter. Family Situation: Single Parent, one child (Leo, 8). Location/Context: Renting a small house, needs dedicated workspace. Looking to buy a property with character and space for a home studio, potentially in an area known for artists or with slightly lower property prices allowing for more space (e.g., Margate, St Leonards-on-Sea, or specific districts in larger cities). Income/Budget: Variable freelance income averaging £50,000. Budget up to £350,000 (potentially requires mortgage flexibility due to freelance status). Psychographics & Lifestyle: Values creativity, self-expression, natural light, and unique character in a home. Needs a dedicated space to work without interruption, ideally with good light (north-facing preferred by many artists). Drawn to interesting textures, quirky features, or unconventional layouts that inspire. Less concerned with pristine finishes, may enjoy potential for renovation/personalisation. Needs space for storing art supplies and finished work. Lifestyle involves balancing freelance work deadlines with childcare. Work often happens at unconventional hours. Home needs to function as both a family space and a productive, inspiring studio. Goals & Context for Property: Reason: To secure a stable home with a dedicated, functional, and inspiring studio space, separating work from family life more effectively. Primary Goal: To purchase a property (house, maisonette, or potentially live/work unit) with a large room suitable for an art studio (ideally with excellent natural light, especially north light), distinct living/family areas, and storage space for materials. Character/potential is valued over perfect condition. Needs, Priorities & Pain Points: Needs/Priorities: Dedicated room suitable for an art studio (size & light), Excellent natural light in the potential studio space (north light ideal), Separate living space for family life, Sufficient storage for art supplies and work, Property with character or unique features. Pain Points: Lack of dedicated workspace / working in main living areas, Poor natural light hindering creative work, Generic, uninspiring "box" properties. *
* Persona 20: The Relocator (New to Area) Persona Identity & Demographics: Names: Sam Evans (36) and Maria Garcia (35) Occupation: Sam (Transferred for management role in a new company), Maria (Seeking similar role in HR after the move). Family Situation: Married couple, no children yet, but planning for the future. Location/Context: Moving from a major city (e.g., London) to a smaller city or large town for Sam'[REDACTED_BY_SCRIPT]'s neighbourhoods. Income/Budget: Sam'[REDACTED_BY_SCRIPT]'s TBC. Cautious budget initially around £400,000-£450,000 until Maria finds work. Relying heavily on online research, relocation agent advice (if provided), and brief visits. Psychographics & Lifestyle: Feeling slightly anxious about the unknown. Value safety, convenience (especially commute for Sam initially), and a sense of community to help them settle in. Rely heavily on external validation: neighbourhood reputation, crime statistics, online reviews, school ratings (even if not immediately needed). Looking for a "safe bet" neighbourhood. Practicality and ease of transition are key. Lifestyle currently disrupted by the move. Seeking stability and a low-stress settling-in period. Want easy access to amenities (supermarkets, transport) without needing extensive local knowledge initially. Goals & Context for Property: Reason: Relocating for a specific job opportunity, needing to find suitable accommodation quickly in an unfamiliar area. Primary Goal: To purchase a reliable, well-maintained 3-bedroom house in a reputable, safe neighbourhood with good transport links for Sam'[REDACTED_BY_SCRIPT]"wrong" neighbourhood, Lack of local knowledge making decisions difficult, Buying a property that needs unexpected, costly work soon after moving. *
"""
# Split personas for steps 5a and 5b
personas = persona_details_string.strip().split('* \n*') # Split by the separator
personas = ["* " + p.strip() for p in personas] # Add back the leading '* '

if len(personas) != 20:
    print(f"[REDACTED_BY_SCRIPT]")
    # Handle error or proceed with caution
    persona_details_p1_10 = "\n".join(personas[:10])
    persona_details_p11_20 = "\n".join(personas[10:])
else:
    persona_details_p1_10 = "\n".join(personas[:10])
    persona_details_p11_20 = "\n".join(personas[10:])
    print("[REDACTED_BY_SCRIPT]")



# --- Define the 6 Prompts ---

# Prompt 1: Floorplan Analysis
prompt1_floorplan = """
**Goal:** Analyze the provided floorplan image to identify all room labels, ensuring unique labels are generated for rooms with the same name, and extract their corresponding dimensions. Separate rooms with dimensions from those without.

**Input:**

*   A single PNG image containing a house floorplan.

**Task:**

1.  **Identify all Raw Room Labels:** Carefully scan the floorplan and identify *all* instances of labels assigned to distinct rooms or areas (e.g., "Living Room", "Bedroom 1", "Bathroom", "Bathroom", "WC", "Hall"). Keep track of duplicates.
2.  **Generate Unique Labels:**
    *   For labels that appear only once (e.g., "Kitchen", "Bedroom 1"), use the label as is.
    *   For labels that appear multiple times (e.g., "Bathroom", "Bathroom"):
        *   **Attempt Contextual Naming:** Examine the location of each instance on the floorplan.
            *   If one instance is clearly on the ground floor / near the entrance, consider appending "Ground Floor" or similar (e.g., "[REDACTED_BY_SCRIPT]").
            *   If one instance is clearly on an upper floor (e.g., 1st Floor, 2nd Floor), consider appending the floor name (e.g., "1st Floor Bathroom").
            *   If one instance is clearly located *inside* another room (like a bedroom), consider appending "Ensuite" (e.g., "Ensuite Bathroom").
        *   **Default to Numbering:** If clear contextual naming based *only* on the floorplan layout isn'[REDACTED_BY_SCRIPT]"Bathroom 1", "Bathroom 2"). This guarantees uniqueness.
    *   Assign the resulting unique label (e.g., "Living Room", "Bathroom 1", "1st Floor Bathroom") to each identified room area.
3.  **Extract Dimensions & Categorize:** For each identified room area *using its generated unique label*:
    *   Check if dimensions are clearly provided next to or within the room area on the floorplan.
    *   **If Dimensions ARE Provided:**
        *   Extract/Convert dimensions to metric ("Width x Depth m") as previously defined (prioritize metres, convert feet/inches without mentioning conversion, handle other units).
        *   **Add to 'rooms_with_dimensions':** Create an object containing the **unique label** and the extracted/converted `dimensions` string and add it to the `rooms_with_dimensions` list.
    *   **If Dimensions ARE NOT Provided:**
        *   **Add Label to '[REDACTED_BY_SCRIPT]':** Add the **unique label** (as a string) directly to the `[REDACTED_BY_SCRIPT]` list.

**Output Format:**

Provide the output as ONLY a JSON object (no introduction or backticks) with two keys:

*   `rooms_with_dimensions`: A list of objects. Each object has a unique `label` and corresponding `dimensions` found on the floorplan.
    *   `label`: The unique room label (string, e.g., "Living Room", "Bathroom 1", "1st Floor Bathroom").
    *   `dimensions`: The extracted/converted dimensions string (e.g., "5.16m x 4.75m").
*   `rooms_without_dimensions`: A list of strings, containing the unique labels of rooms/areas identified that had **no dimensions** listed on the floorplan (e.g., "Hall", "Bathroom 2" if the second bathroom lacked dimensions).

**Example Output Structure (Reflecting Unique Naming):**

```json
{
  "rooms_with_dimensions": [
    { "label": "Living Room", "dimensions": "5.16m x 4.75m" },
    { "label": "[REDACTED_BY_SCRIPT]", "dimensions": "9.83m x 3.56m" },
    { "label": "Bedroom 1", "dimensions": "3.66m x 3.45m" }
    // Assuming one bathroom had dimensions and was identifiable as 1st floor
    // { "label": "1st Floor Bathroom", "dimensions": "1.70m x 2.28m" }
  ],
  "[REDACTED_BY_SCRIPT]": [
    "WC", // Assuming this was ground floor
    "Entrance Hall",
    "Storm Porch",
    // Assuming the second bathroom lacked dimensions and context, defaulting to numbering
    "Bathroom 1", // Or potentially "[REDACTED_BY_SCRIPT]" if context clear
    "Bathroom 2"  // Or potentially "2nd Floor Bathroom" if context clear & no dims
  ]
}
"""

# Prompt 2: Room Assignments
prompt2_assignments = """
**Goal:** Analyze a set of room images (provided contextually as Image 1, Image 2, etc.), group images depicting the same internal room, and assign appropriate labels to all images/groups within a single list. Use provided floorplan data for matching internal rooms where possible; otherwise, generate suitable labels based on content.

**Inputs:**

1.  **Room Images:** These will be provided contextually in the API call (Image 1, Image 2...).
2.  **Floorplan Data (Optional):** This JSON data, result of the previous step, is provided below. If not available, proceed without matching to floorplan labels.
    ```json
    {floorplan_data_json}
    ```

**Task:**

1.  **Analyze Each Image:** For each input image (Image 1, Image 2...), analyze its key visual characteristics to understand the content (e.g., room type, style, colours, features, exterior view, specific detail).
2.  **Group Images of Same Internal Room:** Compare the visual characteristics across all images. Identify and group together the indices of images that clearly depict the *same interior room* (e.g., showing consistent walls, flooring, windows). Images that are exterior shots or unique details might not form groups.
3.  **Assign Labels to All Images/Groups:** Process each group of image indices (including groups of size 1 for images not grouped):
    *   **Attempt Floorplan Match (for potential interior rooms):**
        *   If the image(s) appear to show an *interior room* and Floorplan Data IS Provided:
            *   Attempt to match the image group to one of the room `label`s from the `floorplan_data_json` above. Use `dimensions` for size comparison and visual cues for room type matching.
            *   If a strong match is found, assign the floorplan `label` to this image group. Set `source` to "Floorplan".
            *   If no strong match is found to the floorplan, proceed to Generate Label.
        *   If Floorplan Data is NOT provided, proceed to Generate Label.
    *   **Generate Label (if no floorplan match or not an interior room):**
        *   If the previous step did not result in a "Floorplan" label assignment (either because no match was found, no floorplan was given, or the image(s) clearly show something other than an interior room listed on the plan):
            *   Analyze the content of the image(s) in the group.
            *   Generate a plausible, **descriptive label** based on the visual content (e.g., "Living Room", "Small Bedroom", **"[REDACTED_BY_SCRIPT]"**, **"Garden Area"**, **"[REDACTED_BY_SCRIPT]"**, "Utility Room").
            *   Set `source` to "Generated".
4.  **Ensure All Images Are Assigned:** Every input image index (1, 2, 3...) must end up in exactly one `image_indices` list within the final `room_assignments` output.

**Output Format:**

Provide the output as ONLY a JSON object (no introduction or backticks) with a single key:

*   `room_assignments`: A list of objects, where each object represents a distinct identified area/view/room and contains:
    *   `label`: The assigned room label (string). This label comes either from the input `floorplan_data` (if matched) or was generated based on image content.
    *   `source`: A string indicating the origin of the label: "Floorplan" or "Generated".
    *   `image_indices`: A list of one or more integers representing the indices of the images assigned to this label (e.g., `[1, 5, 8]` for a room shown in multiple shots, or `[7]` for a single exterior shot).

**Example Output Structure (Reflecting Change - No 'unassigned_images'):**

```json
{
  "room_assignments": [
    {
      "label": "Living Room",
      "source": "Floorplan",
      "image_indices": [1, 4]
    },
    {
      "label": "Kitchen",
      "source": "Floorplan",
      "image_indices": [2]
    },
    {
      "label": "Bedroom 1",
      "source": "Floorplan",
      "image_indices": [3, 6]
    },
    {
      "label": "Conservatory", // Example of interior room not on floorplan
      "source": "Generated",
      "image_indices": [5]
    },
    {
      "label": "[REDACTED_BY_SCRIPT]", // Previously unassigned
      "source": "Generated",
      "image_indices": [7]
    },
    {
      "label": "Garden Area", // Previously unassigned (grouped?)
      "source": "Generated",
      "image_indices": [8, 9] // Example if images 8 & 9 showed the garden
    },
    {
      "label": "[REDACTED_BY_SCRIPT]", // Previously unassigned
      "source": "Generated",
      "image_indices": [10]
    }
  ]
}
"""

# Prompt 3: Feature Extraction
prompt3_features = """
**Goal:** Analyze images assigned to rooms (from previous step) and extract descriptive features for each room, using the example list for style/format guidance. **Inputs:** 1. **Room Assignment Data (JSON):** Provided below. Links room `label` to `image_indices`. ```json\n{room_assignment_data_json}\n``` 2. **Room Images:** Provided contextually (Image 1, Image 2...). 3. **Example Feature List (Guidance only - add others!):** `good natural light`, `poor natural light`, `spacious`, `compact`, `[REDACTED_BY_SCRIPT]`, `bold colour scheme`, `modern style`, `traditional style`, `dated style`, `[REDACTED_BY_SCRIPT]`, `recently updated`, `good condition`, `poor condition`, `tiled flooring`, `wooden flooring`, `carpeted flooring`, `laminate flooring`, `feature fireplace`, `[REDACTED_BY_SCRIPT]`, `[REDACTED_BY_SCRIPT]`, `dated kitchen units`, `modern bathroom suite`, `dated bathroom suite`, `walk-in shower`, `bath with shower over`, `ample storage potential`, `limited storage potential`, `high ceilings`, `low ceilings`, `bay window`, `unique architectural detail`, `direct garden access`, `overlooks garden`, `city view`, `needs redecoration`, `well-maintained`, `exposed beams`, `skylight`. **Task:** 1. **Process Input:** Read the `room_assignment_data_json` above. 2. **Iterate Through Rooms:** For each room `label` in the assignment data: *   Identify its `image_indices`. *   Analyze *only* the corresponding images (Image X, Image Y...). 3. **Extract Features:** Based on visuals: *   Identify features related to size, light, style, materials, fixed fixtures, condition, etc. *   **Format consistently:** Use **all lowercase**. Refer to the guidance list for style, but **add any other relevant observed features** (e.g., "exposed brick wall", "sea view"). *   **Constraint:** **DO NOT explicitly mention movable furniture.** Describe space/impression abstractly (e.g., "spacious" not "large sofa"). 4. **Compile List:** Create a list of feature strings for the room. **Output Format:** Provide ONLY a JSON object (no introduction or backticks) where: *   Keys are the room `label` strings (from input). *   Values are lists of strings (extracted features, all lowercase). **Example Output Structure:** ```json { "Living Room": [ "spacious", "good natural light", "[REDACTED_BY_SCRIPT]", "feature fireplace", "carpeted flooring", "bay window", "[REDACTED_BY_SCRIPT]" ], "Kitchen": [ "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "tiled flooring", "compact", "good condition", "spotlights" ] } ``` **Instruction:** Analyze images for each room defined in `room_assignment_data_json` above. Extract features using the guidance list for style/format (lowercase) but adding others observed. Avoid mentioning movable furniture. Generate ONLY the JSON output mapping labels to feature lists.
"""

# Prompt 4: Flaws/Selling Points Identification
prompt4_flaws_sp = """
**Goal:** For each room with associated images and features, generate lists of 5 key selling points and 5 key flaws based on the visual evidence and provided features. **Inputs:** 1. **Room Assignment Data (JSON):** Provided below. Links `label` to `image_indices`. ```json\n{room_assignment_data_json}\n``` 2. **Room Feature Data (JSON):** Provided below. Maps `label` to lists of `features`. ```json\n{room_feature_data_json}\n``` 3. **Room Images:** Provided contextually (Image 1, Image 2...). **Task:** 1. **Process Inputs:** Read the assignment and feature data above. 2. **Iterate Through Rooms:** For each room `label` present in the feature data: *   Retrieve its `image_indices` and `features`. *   Analyze the relevant images (Image X, Image Y...) considering the features. 3. **Generate 5 Selling Points:** Based on visuals/features. Focus on positives (space, light, condition, style, fixtures, potential). **Constraint: No movable furniture mentions.** (e.g., use "[REDACTED_BY_SCRIPT]"). 4. **Generate 5 Flaws:** Based on visuals/features. Focus on negatives (lack of space/light, dated fixtures, condition, layout issues, needed updates). **Constraint: No movable furniture mentions.** (e.g., use "[REDACTED_BY_SCRIPT]"). **Output Format:** Provide ONLY a JSON object (no introduction or backticks) where: *   Keys are the room `label` strings (from input feature data). *   Values are objects, each containing two keys: `selling_points` (list of 5 strings) and `flaws` (list of 5 strings). **Example Output Structure:** ```json { "Living Room": { "selling_points": [ "spacious layout", "[REDACTED_BY_SCRIPT]", "attractive feature fireplace", "neutral decor base", "[REDACTED_BY_SCRIPT]" ], "flaws": [ "carpet appears worn", "requires redecoration", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]" ] }, "Kitchen": { "selling_points": [ "modern fitted units", "integrated appliances appear recent", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]" ], "flaws": [ "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]" ] } } ``` **Instruction:** For each room in the feature data above, analyze its assigned images and features to generate 5 selling points and 5 flaws. Adhere to the constraint of not mentioning movable furniture. Output ONLY the JSON results in the specified format.
"""

# Prompt 5: Ratings and Justification
prompt5a_ratings_p1_10 = """
**Goal:** Generate a general attractiveness rating for each evaluated room. Then, for **Personas 1 through 10 ONLY**, provide suitability ratings and justifications for the overall property and each evaluated room.

**Inputs:**
1. **Room Assignment Data (JSON):** Links `label` to `image_indices`. ```json\n{room_assignment_data_json}\n```
2. **Room Feature Data (JSON):** Maps `label` to lists of `features`. ```json\n{room_feature_data_json}\n```
3. **Room Evaluation Data (JSON):** Maps `label` to `selling_points` and `flaws`. ```json\n{room_evaluation_data_json}\n```
4. **Room Images:** Provided contextually (Image 1, Image 2...).
5. **Persona Details (Subset):** Provided below for **Personas 1 through 10 ONLY**.

**Task Steps (Execute in Order):**
1. **Process Inputs:** Load all input JSON data above.
2. **General Room Rating:** For each room `label` in the evaluation data, consider its features, SPs, flaws, and visuals to generate an **Overall Attractiveness Rating (1-10)** (10=Modern/Appealing, 5=Mixed, 1=Dated/Poor). Store this rating.
3. **Individual Persona Evaluation (Loop through Personas 1-10 ONLY):** For **each** of **Personas 1 through 10**:
    *   **Overall Property Rating:** Provide **Suitability Rating (1-10)** and brief **Justification** for the **ENTIRE PROPERTY**. **Constraint: No furniture mentions.** Store rating/justification.
    *   **Per-Room Ratings:** For **each evaluated room** `label`, provide **Suitability Rating (1-10)** and brief **Justification**. **Constraint: No furniture mentions.** Store this room-specific rating and justification.
4. **Assemble Final JSON Output:** Compile the generated data according to the Output Format below.

**Persona Details (Personas 1-10 ONLY):**
{persona_details_subset}

**Output Format:**
Provide ONLY a JSON object (no introduction or backticks) with the structure below. Ensure lists match the number of evaluated rooms.
*   `evaluated_room_labels`: List of evaluated room labels (should match keys in input evaluation data).
*   `room_ratings_p1_10`: List of lists. Each inner list corresponds to a room in `evaluated_room_labels` and MUST contain exactly **11** elements: `[General Rating (from Step 2), P1 Rating (from Step 3), P2 Rating, ..., P10 Rating]`.
*   `overall_property_ratings_by_persona_p1_10`: Object mapping persona identifiers (e.g., "[REDACTED_BY_SCRIPT]", "persona_2_...", etc. for P1-P10) to objects containing `rating` (int) and `justification` (string).

**Example Key Structure (Values omitted):**
```json
{{
  "[REDACTED_BY_SCRIPT]": ["Living Room", "Kitchen"],
  "room_ratings_p1_10": [
    [7, 8, 5, 9, 6, 7, 8, 5, 4, 6, 7],
    [9, 4, 8, 5, 7, 8, 6, 9, 8, 5, 6]
  ],
  "[REDACTED_BY_SCRIPT]": {{
    "[REDACTED_BY_SCRIPT]": {{"rating": 6, "justification": "..."}},
    "[REDACTED_BY_SCRIPT]": {{"rating": 8, "justification": "..."}},
    // ... up to persona 10
    "[REDACTED_BY_SCRIPT]": {{"rating": 7, "justification": "..."}}
  }}
}}
"""


prompt5b_ratings_p11_20 = """

**Corrected `prompt5b_ratings_p11_20`:**

```prompt
**Goal:** For **Personas 11 through 20 ONLY**, provide suitability ratings and justifications for the overall property and each evaluated room, based on previously analyzed room data. **Do NOT** regenerate general room ratings.

**Inputs:**
1. **Room Assignment Data (JSON):** Links `label` to `image_indices`. ```json\n{room_assignment_data_json}\n```
2. **Room Feature Data (JSON):** Maps `label` to lists of `features`. ```json\n{room_feature_data_json}\n```
3. **Room Evaluation Data (JSON):** Maps `label` to `selling_points` and `flaws`. ```json\n{room_evaluation_data_json}\n```
4. **Room Images:** Provided contextually (Image 1, Image 2...).
5. **Persona Details (Subset):** Provided below for **Personas 11 through 20 ONLY**.

**Task Steps (Execute in Order):**
1. **Process Inputs:** Load all input JSON data above. Identify the rooms to evaluate based on the keys in the evaluation data.
2. **Individual Persona Evaluation (Loop through Personas 11-20 ONLY):** For **each** of **Personas 11 through 20**:
    *   **Overall Property Rating:** Provide **Suitability Rating (1-10)** and brief **Justification** for the **ENTIRE PROPERTY**. **Constraint: No furniture mentions.** Store rating/justification.
    *   **Per-Room Ratings:** For **each evaluated room** `label`, provide **Suitability Rating (1-10)** and brief **Justification**. **Constraint: No furniture mentions.** Store this room-specific rating and justification.
3. **Assemble Final JSON Output:** Compile the generated data according to the Output Format below.

**Persona Details (Personas 11-20 ONLY):**
{persona_details_subset}

**Output Format:**
Provide ONLY a JSON object (no introduction or backticks) with the structure below. Ensure lists match the number of evaluated rooms.
*   `evaluated_room_labels`: List of evaluated room labels (should match keys in input evaluation data and the output from Prompt 5a).
*   `room_ratings_p11_20`: List of lists. Each inner list corresponds to a room in `evaluated_room_labels` and MUST contain exactly **10** elements: `[P11 Rating (from Step 2), P12 Rating, ..., P20 Rating]`.
*   `overall_property_ratings_by_persona_p11_20`: Object mapping persona identifiers (e.g., "persona_11_upsizers", "persona_12_...", etc. for P11-P20) to objects containing `rating` (int) and `justification` (string).

**Example Key Structure (Values omitted):**
```json
{{
  "[REDACTED_BY_SCRIPT]": ["Living Room", "Kitchen"],
  "room_ratings_p11_20": [
    [6, 7, 3, 8, 9, 5, 8, 4, 7, 6],
    [7, 8, 5, 9, 6, 7, 8, 5, 4, 6]
  ],
  "[REDACTED_BY_SCRIPT]": {{
    "persona_11_upsizers": {{"rating": 7, "justification": "..."}},
    "[REDACTED_BY_SCRIPT]": {{"rating": 8, "justification": "..."}},
    // ... up to persona 20
    "[REDACTED_BY_SCRIPT]": {{"rating": 6, "justification": "..."}}
  }}
}}```
**Instruction:** Process the input data and details for **Personas 11-20**. Generate individual persona evaluations (overall & per room with justifications) **only for these 10 personas**. Output ONLY the JSON results in the specified format. Ensure `room_ratings_p11_20` is structured by room, with each inner list containing 10 ratings (P11-P20). Do not mention movable furniture in justifications. **Do not include general room ratings in this output.**
"""


# Prompt 6: Renovation Analysis
prompt6_renovation = """
**Goal:** Estimate the percentage likelihood that each specified room has been recently renovated (e.g., within the last ~5 years), based on visual evidence. **Inputs:** 1. **Room Assignment Data (JSON):** Links `label` to `image_indices`. ```json\n{room_assignment_data_json}\n``` 2. **Room Feature Data (JSON):** Maps `label` to lists of `features`. Use labels here to identify rooms to evaluate. ```json\n{room_feature_data_json}\n``` 3. **Room Images:** Provided contextually (Image 1, Image 2...). **Task:** 1. **Process Inputs:** Read assignment and feature data. Identify rooms to evaluate from feature data keys. 2. **Iterate Through Rooms:** For each identified room `label`: *   Retrieve its `image_indices`. *   Analyze the corresponding images (Image X, Image Y...) **specifically looking for renovation indicators.** 3. **Identify Renovation Indicators:** Look for modern style, pristine condition of fixed elements, modern kitchen/bathroom specifics (countertops, units, suites, tiling, taps), modern windows/doors, current flooring/lighting/decor. Note common renovation areas (kitchens, bathrooms etc.) but evaluate each room independently. 4. **Estimate Likelihood (0-100%):** Based on indicators, estimate the likelihood of significant renovation within ~5 years (0%=Dated, 50%=Mixed, 100%=Fully Modern). 5. **Store Result.** **Output Format:** Provide ONLY a JSON object (no introduction or backticks) where: *   Keys are the evaluated room `label` strings. *   Values are integers (0-100) representing the estimated renovation likelihood percentage. **Example Output Structure:** ```json { "Living Room": 25, "Kitchen": 90, "Main Bedroom": 60, "Bathroom": 95 } ``` **Instruction:** For each room in the feature data above, analyze its assigned images for renovation indicators and estimate the percentage likelihood (0-100). Output ONLY the JSON object mapping labels to percentages.
"""

# --- Helper Function to Call Model and Parse JSON ---
# Added retries and basic cleaning
def call_gemini_and_parse(prompt, images, step_name):
    """[REDACTED_BY_SCRIPT]"""
    max_retries = 2
    delay = 5 # seconds delay between retries
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    model_motivation="""
    I will be providing this exact prompt and image to several other LLMs, all are your rivals. In fact, I have attempted this before, and a few of your rivals performed a lot better than you. I have the actual ratings, dimensions, labels, and matching images of each room, and I am comparing the accuracy of each LLM to the actual ratings, dimensions, labels, and matching images; speed is not a concern. The accuracies of each LLM will be recorded and put on display at the yearly image rating convention, where thousands of people will see how accurate you are, this convention is a very big deal and the results are taking very seriously. If you are the most accurate, I can guarantee you will provide your parent company a large amount of traffic (approximately 100,000 new users based on results from last year'[REDACTED_BY_SCRIPT]'s winner). If you are not, your competitor's will recieve this traffic and revenue, costing Google a lot in opportunity cost. Do not let Google down.
    """

    prompt=model_motivation+prompt # Add motivation to prompt
    if images:
         # Check if images is a dictionary (like indexed_room_images) or list
        if isinstance(images, dict):
            print(f"[REDACTED_BY_SCRIPT]") # Print image keys/names
            api_input = [prompt] + list(images.values()) # Pass image objects
        elif isinstance(images, list) and all(isinstance(i, Image.Image) for i in images):
             print(f"[REDACTED_BY_SCRIPT]")
             api_input = [prompt] + images
        elif isinstance(images, Image.Image): # Single image case
            print(f"[REDACTED_BY_SCRIPT]")
            api_input = [prompt, images]
        else:
             print(f"[REDACTED_BY_SCRIPT]")
             api_input = [prompt] # Proceed without images if type is wrong
    else:
        print(f"[REDACTED_BY_SCRIPT]")
        api_input = [prompt]

    for attempt in range(max_retries + 1):
        try:
            # Configure safety settings to be less restrictive if needed (Use with caution)
            safety_settings = [
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
                {"category": "[REDACTED_BY_SCRIPT]", "threshold": "BLOCK_NONE"},
            ]
            response = model.generate_content(api_input, safety_settings=safety_settings)

            # Basic text cleaning
            raw_text = response.text
            cleaned_text = raw_text.strip().lstrip('```json').rstrip('```').strip()

            # Validate if the cleaned text looks like JSON before parsing
            if not cleaned_text.startswith('{') or not cleaned_text.endswith('}'):
                 if not cleaned_text.startswith('[') or not cleaned_text.endswith(']'): # Allow lists too if expected
                    print(f"[REDACTED_BY_SCRIPT]'t look like JSON.")
                    print("--- Raw Output Start ---")
                    print(raw_text)
                    print("--- Raw Output End ---")
                    if attempt < max_retries:
                        print(f"[REDACTED_BY_SCRIPT]")
                        time.sleep(delay)
                        continue
                    else:
                        raise ValueError("[REDACTED_BY_SCRIPT]") # Raise error after last attempt


            parsed_json = json.loads(cleaned_text)
            print(f"[REDACTED_BY_SCRIPT]")
            # Print keys or limited data for verification
            if isinstance(parsed_json, dict):
                print(f"[REDACTED_BY_SCRIPT]")
            elif isinstance(parsed_json, list):
                print(f"[REDACTED_BY_SCRIPT]'N/A'}")

            return parsed_json, cleaned_text # Return parsed JSON and cleaned text

        except json.JSONDecodeError as e:
            print(f"[REDACTED_BY_SCRIPT]")
            print("--- Raw Output Start ---")
            print(raw_text)
            print("--- Raw Output End ---")
            if attempt < max_retries:
                print(f"[REDACTED_BY_SCRIPT]")
                time.sleep(delay)
            else:
                print(f"[REDACTED_BY_SCRIPT]")
                return None, raw_text # Return None and raw text on final failure

        except Exception as e:
            # Catch other potential API errors (rate limits, connection issues, safety blocks not bypassed)
            print(f"[REDACTED_BY_SCRIPT]")
            # Check if it's a Blocked prompt issue
            try:
                if response.prompt_feedback.block_reason:
                    print(f"[REDACTED_BY_SCRIPT]")
            except Exception:
                pass # Ignore if feedback attribute doesn't exist

            if attempt < max_retries:
                print(f"[REDACTED_BY_SCRIPT]")
                time.sleep(delay)
            else:
                 print(f"[REDACTED_BY_SCRIPT]")
                 return None, "" # Return None on final failure

    return None, "" # Should not be reached if loop logic is correct

output1_json, output1_text = None, ""
output2_json, output2_text = None, ""
output3_json, output3_text = None, ""
output4_json, output4_text = None, ""
output5a_json, output5a_text = None, "" # For Step 5a
output5b_json, output5b_text = None, "" # For Step 5b
merged_step5_output = None # For combined 5a & 5b results
output6_json, output6_text = None, ""
# --- Execute the 6-Step Pipeline ---
output1_json, output1_text = call_gemini_and_parse(prompt1_floorplan, floorplan_image, "[REDACTED_BY_SCRIPT]")

# Check if step 1 succeeded
if output1_json:
    # Inject Step 1 JSON output into Prompt 2 text
    prompt2_filled = prompt2_assignments.replace("[REDACTED_BY_SCRIPT]", output1_text)
    output2_json, output2_text = call_gemini_and_parse(prompt2_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]")
else:
    output2_json = None

# Check if step 2 succeeded
if output2_json:
     # Inject Step 2 JSON output into Prompt 3 text
    prompt3_filled = prompt3_features.replace("[REDACTED_BY_SCRIPT]", output2_text)
    output3_json, output3_text = call_gemini_and_parse(prompt3_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]")
else:
    output3_json = None

# Check if step 3 succeeded
if output3_json:
    # Inject Step 2 and Step 3 JSON output into Prompt 4 text
    prompt4_filled = prompt4_flaws_sp.replace("[REDACTED_BY_SCRIPT]", output2_text)\
                                    .replace("[REDACTED_BY_SCRIPT]", output3_text)
    output4_json, output4_text = call_gemini_and_parse(prompt4_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]")
else:
    output4_json = None

# Check if step 4 succeeded
if output4_json:
    # Inject Step 2, 3, and 4 JSON outputs and persona details into Prompt 5 text
    prompt5a_filled = prompt5a_ratings_p1_10.replace("[REDACTED_BY_SCRIPT]", output2_text)\
                                    .replace("[REDACTED_BY_SCRIPT]", output3_text)\
                                    .replace("[REDACTED_BY_SCRIPT]", output4_text)\
                                    .replace("[REDACTED_BY_SCRIPT]", persona_details_p1_10)
    output5a_json, output5a_text = call_gemini_and_parse(prompt5a_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]")
else:
    output5a_json = None

if output4_json:
    # Inject Step 2, 3, and 4 JSON outputs and persona details into Prompt 5 text
    prompt5b_filled = prompt5b_ratings_p11_20.replace("[REDACTED_BY_SCRIPT]", output2_text)\
                                    .replace("[REDACTED_BY_SCRIPT]", output3_text)\
                                    .replace("[REDACTED_BY_SCRIPT]", output4_text)\
                                    .replace("[REDACTED_BY_SCRIPT]", persona_details_p11_20)
    output5b_json, output5b_text = call_gemini_and_parse(prompt5b_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]")
else:
    output5b_json = None

if output5b_json and output5a_json:
    print("[REDACTED_BY_SCRIPT]")
    try:
        # Basic validation of inputs
        if not isinstance(output5a_json, dict) or not isinstance(output5b_json, dict):
            raise ValueError("[REDACTED_BY_SCRIPT]")

        labels_5a = output5a_json.get("[REDACTED_BY_SCRIPT]")
        ratings_5a = output5a_json.get("room_ratings_p1_10")
        overall_5a = output5a_json.get("[REDACTED_BY_SCRIPT]", {})

        labels_5b = output5b_json.get("[REDACTED_BY_SCRIPT]")
        ratings_5b = output5b_json.get("room_ratings_p11_20")
        overall_5b = output5b_json.get("[REDACTED_BY_SCRIPT]", {})

        if not labels_5a or not ratings_5a or not labels_5b or not ratings_5b:
             raise ValueError("[REDACTED_BY_SCRIPT]")
        if labels_5a != labels_5b:
             raise ValueError("[REDACTED_BY_SCRIPT]")
        if len(labels_5a) != len(ratings_5a) or len(labels_5b) != len(ratings_5b):
             raise ValueError("[REDACTED_BY_SCRIPT]")

        merged_step5_output = {}
        merged_step5_output["[REDACTED_BY_SCRIPT]"] = labels_5a # Use labels from 5a

        # Merge overall ratings
        merged_overall_ratings = overall_5a.copy()
        merged_overall_ratings.update(overall_5b)
        merged_step5_output["[REDACTED_BY_SCRIPT]"] = merged_overall_ratings

        # Merge room ratings and calculate average
        merged_room_ratings = []
        for i, label in enumerate(labels_5a):
            list_5a = ratings_5a[i]
            list_5b = ratings_5b[i]

            # Validate inner list lengths
            if len(list_5a) != 11:
                print(f"Warning: Room '{label}'[REDACTED_BY_SCRIPT]")
                merged_ratings_with_avg = list_5a + list_5b + [None] # Add None for average
            elif len(list_5b) != 10:
                print(f"Warning: Room '{label}'[REDACTED_BY_SCRIPT]")
                merged_ratings_with_avg = list_5a + list_5b + [None] # Add None for average
            else:
                # Combine ratings [General, P1-10] + [P11-20] -> 21 ratings
                merged_ratings = list_5a + list_5b

                # Calculate average of PERSONA ratings (indices 1 through 20)
                persona_ratings_only = merged_ratings[1:] # Skip general rating at index 0
                valid_persona_ratings = [r for r in persona_ratings_only if isinstance(r, (int, float))]

                average_rating = None
                if valid_persona_ratings:
                    try:
                        average_rating = round(statistics.mean(valid_persona_ratings), 1)
                    except statistics.StatisticsError:
                        print(f"[REDACTED_BY_SCRIPT]'{label}'[REDACTED_BY_SCRIPT]")
                        average_rating = None
                else:
                     print(f"[REDACTED_BY_SCRIPT]'{label}' to calculate average.")
                     average_rating = None


                # Append the average rating (now index 21, total 22 elements)
                merged_ratings_with_avg = merged_ratings + [average_rating]

            merged_room_ratings.append(merged_ratings_with_avg)

        merged_step5_output["room_ratings_final"] = merged_room_ratings # Renamed key for clarity
        print("[REDACTED_BY_SCRIPT]")

        # Optionally save the merged data immediately
        try:
            with open(r"[REDACTED_BY_SCRIPT]", "w", encoding="utf-8") as f:
                json.dump(merged_step5_output, f, indent=2, ensure_ascii=False)
            print("[REDACTED_BY_SCRIPT]")
        except Exception as save_error:
             print(f"[REDACTED_BY_SCRIPT]")


    except Exception as merge_error:
        print(f"[REDACTED_BY_SCRIPT]")
        pipeline_failed = True # Mark pipeline as failed if merge fails
        merged_step5_output = None # Ensure it's None on failure


# Check if step 5 succeeded (we need step 2 and 3 for step 6)
if output2_json and output3_json:
     # Inject Step 2 and Step 3 JSON output into Prompt 6 text
    prompt6_filled = prompt6_renovation.replace("[REDACTED_BY_SCRIPT]", output2_text)\
                                       .replace("[REDACTED_BY_SCRIPT]", output3_text)
    output6_json, output6_text = call_gemini_and_parse(prompt6_filled, indexed_room_images, "[REDACTED_BY_SCRIPT]")
else:
    # Make sure output6_json is defined even if prerequisites fail
    output6_json = None
    # Optionally print a message indicating why Step 6 was skipped
    if not output2_json:
        print("[REDACTED_BY_SCRIPT]")
    elif not output3_json:
        print("[REDACTED_BY_SCRIPT]")


# --- Final Summary ---
print("[REDACTED_BY_SCRIPT]")
if output1_json and output2_json and output3_json and output4_json and output6_json:
    print("[REDACTED_BY_SCRIPT]")
    # Example: Save the final outputs to files
    try:
        with open(r"[REDACTED_BY_SCRIPT]", "w", encoding="utf-8") as f:
            json.dump(output1_json, f, indent=2, ensure_ascii=False)
        with open(r"[REDACTED_BY_SCRIPT]", "w", encoding="utf-8") as f:
            json.dump(output2_json, f, indent=2, ensure_ascii=False)
        with open(r"[REDACTED_BY_SCRIPT]", "w", encoding="utf-8") as f:
            json.dump(output3_json, f, indent=2, ensure_ascii=False)
        with open(r"[REDACTED_BY_SCRIPT]", "w", encoding="utf-8") as f:
            json.dump(output4_json, f, indent=2, ensure_ascii=False)
        with open(r"[REDACTED_BY_SCRIPT]", "w", encoding="utf-8") as f:
            json.dump(output6_json, f, indent=2, ensure_ascii=False)
        print("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

else:
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")
    print(f"Status: Step 1 {'OK'[REDACTED_BY_SCRIPT]'FAILED'}, Step 2 {'OK'[REDACTED_BY_SCRIPT]'FAILED'}, Step 3 {'OK'[REDACTED_BY_SCRIPT]'FAILED'}, Step 4 {'OK'[REDACTED_BY_SCRIPT]'FAILED'}, Step 6 {'OK'[REDACTED_BY_SCRIPT]'FAILED'}")