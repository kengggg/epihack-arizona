# Contextual Risk Data — Data Sources

## Human (33 sources)

| Type | Source | Description (from notes) | URL | Status | Data Available | Machine Readable | Online | Update frequency | Data free to use | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Human | [MEDSIS](https://azdhs.gov/medsis/) | AZ state reportable disease system (~88 conditions, confirmed/probable cases). Named "the backbone and the bottleneck" in Info Flow table. | [link](https://azdhs.gov/medsis/) | Closed | ✗ | ✗ | ✗ | ✗ | ✗ | MEDSIS is access-restricted via user agreement; not public open data |
| Human | [BioSense Platform (NSSP)](https://www.cdc.gov/nssp/php/about/about-nssp-and-the-biosense-platform.html) | CDC syndromic surveillance — ED visits and discharge data. Named alongside MEDSIS as the two paradigms in tension. | [link](https://www.cdc.gov/nssp/php/about/about-nssp-and-the-biosense-platform.html) | Closed | ✗ | ✗ | ✗ | ✗ | ✗ | BioSense requires site-admin onboarding; no public data |
| Human | [ESSENCE](https://www.cdc.gov/nssp/php/onboarding-toolkits/essence.html) | Analytic tool sitting on BioSense; "medical record data" in Wildlife/VBD table. | [link](https://www.cdc.gov/nssp/php/onboarding-toolkits/essence.html) | Closed | ✗ | ✗ | ✗ | ✗ | ✗ | ESSENCE has API but only for authorized BioSense users |
| Human | [ASIIS (notes: "AZIIS")](https://www.azdhs.gov/preparedness/epidemiology-disease-control/immunization/asiis/index.php) | AZ State Immunization Information System. Gap: doesn't collect adult immunizations. | [link](https://www.azdhs.gov/preparedness/epidemiology-disease-control/immunization/asiis/index.php) | Closed | ✗ | ✗ | ✗ | ✗ | ✗ | ASIIS registry; provider-only access |
| Human | [AZ Cancer Registry](https://www.azdhs.gov/preparedness/public-health-statistics/cancer-registry/index.php) | Population-based cancer surveillance under ADHS. | [link](https://www.azdhs.gov/preparedness/public-health-statistics/cancer-registry/index.php) | Partial | ✓ | ✗ | ✗ | ✓ | ✓ | Has Data Dashboard + PDF Cancer Reports; no raw machine-readable downloads at this URL |
| Human | [PulseNet](https://www.cdc.gov/pulsenet/) | CDC molecular subtyping network for foodborne/genomic disease data. | [link](https://www.cdc.gov/pulsenet/) | Closed | ✗ | ✗ | ✗ | ✗ | ✗ | Lab network; not a public data product |
| Human | [ADHS Vital Records](https://www.azdhs.gov/licensing/vital-records/index.php) | Death and live birth records. | [link](https://www.azdhs.gov/licensing/vital-records/index.php) | Closed | ✗ | ✗ | ✗ | ✗ | ✗ | Personal vital records ordering; no bulk open data here |
| Human | [AZSVI](https://www.azdhs.gov/documents/policy-intergovernmental-affairs/health-equity/azsvi-technical-data-documentation.pdf) | AZ-tailored Social Vulnerability Index (built on CDC/ATSDR SVI). Used for local vaccine work and wastewater. | [link](https://www.azdhs.gov/documents/policy-intergovernmental-affairs/health-equity/azsvi-technical-data-documentation.pdf) | Partial | ✗ | ✗ | ✗ | ✗ | ✓ | URL is technical PDF documentation only; no data file at this URL |
| Human | [CDC/ATSDR SVI](https://www.atsdr.cdc.gov/place-health/php/svi/index.html) | National 16-metric vulnerability index from ACS data (basis for AZSVI). | [link](https://www.atsdr.cdc.gov/place-health/php/svi/index.html) | Open | ✓ | ✓ | ✓ | ✓ | ✓ | CDC/ATSDR SVI provides CSV/SHP downloads, free, updated periodically (~biennial) |
| Human | [US Census / ACS](https://www.census.gov/) | Demographic baseline used across tables. | [link](https://www.census.gov/) | Open | ✓ | ✓ | ✓ | ✓ | ✓ | Census provides public API and bulk downloads; updated regularly |
| Human | [AZ Health Alert Network (AzHAN)](https://han.health.azdhs.gov/) | Provider alerts infrastructure. | [link](https://han.health.azdhs.gov/) | Closed | ✗ | ✗ | ✗ | ✗ | ✗ | AzHAN is login-only secure alert system |
| Human | [HEAT.AZ.gov](https://heat.azdhs.gov/) | ADHS central heat hub — cooling centers, HRI dashboard, HeatRisk. | [link](https://heat.azdhs.gov/) | Partial | ✗ | ✗ | ✗ | ✓ | ✓ | Resource hub aggregating links; not raw open data |
| Human | [Pima Beat the Heat](https://www.pima.gov/2042/Beat-the-Heat) | Cooling center map + heat education, PCHD + City of Tucson. | [link](https://www.pima.gov/2042/Beat-the-Heat) | Info only | ✗ | ✗ | ✗ | ✗ | ✓ | Information page only |
| Human | [Pima Cooling Centers map](https://www.pima.gov/2307/Cooling-Centers) | Interactive map of cooling centers, hydration stations, respite sites. | [link](https://www.pima.gov/2307/Cooling-Centers) | Partial | ✓ | ✗ | ✗ | ✓ | ✓ | Interactive cooling-center map; no obvious bulk download/API at this URL |
| Human | [211 Arizona](https://211arizona.org/) | Statewide info & referral hotline; used for cooling center location and transport. | [link](https://211arizona.org/) | Info only | ✗ | ✗ | ✗ | ✗ | ✓ | Service referral program; no public dataset |
| Human | [MySidewalk](https://www.mysidewalk.com/) | Dashboard platform named in Info Flow table. | [link](https://www.mysidewalk.com/) | Commercial | ✓ | ✓ | ✗ | ✓ | ✗ | Paid SaaS platform |
| Human | [ProMED](https://www.promedmail.org/) | Global outbreak alerts (One Health — human, animal, plant). | [link](https://www.promedmail.org/) | Partial | ✓ | ✗ | ✗ | ✓ | ✓ | Alerts free to read as HTML; bulk API requires paid subscription |
| Human | [Pima County Health Department (PCHD)](https://www.pima.gov/2031/Health) | Named as external-stakeholder data source for rural/tribal table. | [link](https://www.pima.gov/2031/Health) | Info only | ✗ | ✗ | ✗ | ✗ | ✓ | Department homepage; not a dataset |
| Human | [Pima HealthHub](https://www.pima.gov/3258/Health-Hub-A-Resource-for-Providers) | PCHD provider-facing notifications platform. | [link](https://www.pima.gov/3258/Health-Hub-A-Resource-for-Providers) | Info only | ✗ | ✗ | ✗ | ✓ | ✓ | Provider notifications/newsletter; not open data |
| Human | [Pima Health Data Portal](https://www.pimahealthdataportal.org/) | Local data on heat, vaccines, BRFSS, mental health. | [link](https://www.pimahealthdataportal.org/) | Partial | ✓ | ✗ | ✗ | ✓ | ✓ | Indicators portal; typically Excel/PDF report exports, no bulk API |
| Human | Tribal Rabies Investigation (ER bite-data forms) | Cited tribal-led example; 1–2 day turnaround, manual entry. | No public URL — tribal/internal | N/A (no URL) |  |  |  |  |  |  |
| Human | EMS / 911 calls | Linked to census tract; named in Heat table. | Operational data — no single public URL | N/A (no URL) |  |  |  |  |  |  |
| Human | Hospital data (ICD codes → hospitalization & mortality) | Generic clinical data type. | Generic — see ICD reference at CDC | N/A (no URL) |  |  |  |  |  |  |
| Human | EHR (Electronic Health Records) | "From health institutions for investigation." | Generic — no single URL | N/A (no URL) |  |  |  |  |  |  |
| Human | Lab reports / confirmed laboratory tests | Generic input to MEDSIS. | Generic | N/A (no URL) |  |  |  |  |  |  |
| Human | Annual state reports on demographics of risk | Heat table. | Various ADHS reports | N/A (no URL) |  |  |  |  |  |  |
| Human | Civic round-table community posts | Community-input channel. | Local/informal | N/A (no URL) |  |  |  |  |  |  |
| Human | [Suicide and mental health data](https://www.azdhs.gov/prevention/womens-childrens-health/injury-prevention/suicide-prevention/) | Info Flow table. | [link](https://www.azdhs.gov/prevention/womens-childrens-health/injury-prevention/suicide-prevention/) | N - inaccessible | ✗ | ✗ | ✗ | ✗ | ✗ | URL 404 / redirects to search |
| Human | Police department data (gun violence) | Info Flow + rural/tribal tables. | Local PD records | N/A (no URL) |  |  |  |  |  |  |
| Human | Fire department data | Rural/tribal table. | Local FD records | N/A (no URL) |  |  |  |  |  |  |
| Human | [Housing & Urban Development Program data](https://www.hud.gov/) | Unhoused intake; basic info collected during program. | [link](https://www.hud.gov/) | Open | ✓ | ✓ | ✓ | ✓ | ✓ | HUD provides data.hud.gov / HUD User with APIs and bulk downloads |
| Human | [Community Health Needs Assessment (CHNA)](https://www.pima.gov/3650/Healthy-Pima) | TMC sponsorship; community health task force. | [link](https://www.pima.gov/3650/Healthy-Pima) | N - inaccessible | ✗ | ✗ | ✗ | ✗ | ✗ | URL redirects to unrelated climate page; original CHNA page not found |
| Human | Direct data from people / patient recall | Unhoused table — "data quality dependent on people's memory." | Operational | N/A (no URL) |  |  |  |  |  |  |

## Animal (12 sources)

| Type | Source | Description (from notes) | URL | Status | Data Available | Machine Readable | Online | Update frequency | Data free to use | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Animal | [USDA APHIS](https://www.aphis.usda.gov/) | Federal animal health; coordinates with state HDs on zoonotic risks (e.g. avian flu). | [link](https://www.aphis.usda.gov/) | Partial | ✓ | ✓ | ✓ | ✓ | ✓ | APHIS has dashboards and downloads for disease data; URL is general homepage |
| Animal | [AZ Department of Agriculture — Animal Services](https://agriculture.az.gov/animal-services) | State veterinarian, livestock/poultry/dairy/meat/egg inspection. | [link](https://agriculture.az.gov/animal-services) | Info only | ✗ | ✗ | ✗ | ✗ | ✓ | Inspections/services page; no data products |
| Animal | [Pima Animal Care Center (PACC)](https://www.pima.gov/2233/Pima-Animal-Care-Center-PACC) | Open-admission shelter; named in rural/farming table (WNV partnership). | [link](https://www.pima.gov/2233/Pima-Animal-Care-Center-PACC) | Info only | ✗ | ✗ | ✗ | ✗ | ✓ | Animal shelter info page; no open data |
| Animal | [NAHLN (National Animal Health Laboratory Network)](https://www.aphis.usda.gov/labs/about-labs/nahln) | Includes AZ Veterinary Diagnostic Laboratory (Region 4). | [link](https://www.aphis.usda.gov/labs/about-labs/nahln) | N - inaccessible | ✗ | ✗ | ✗ | ✗ | ✗ | URL 404 (NAHLN page moved/removed) |
| Animal | [AZ Rabies Program (animal-side)](https://www.azdhs.gov/preparedness/epidemiology-disease-control/rabies/index.php) | Reactive testing of biting animals (typically dogs). | [link](https://www.azdhs.gov/preparedness/epidemiology-disease-control/rabies/index.php) | Partial | ✓ | ✗ | ✗ | ✓ | ✓ | Has Data, Publications & Maps section (PDFs/maps); not raw machine-readable downloads |
| Animal | Tribal Departments of Agriculture | Some tribes maintain their own; coordinate with USDA + state. | Tribe-specific | N/A (no URL) |  |  |  |  |  |  |
| Animal | Bio-control data for seasonal variability | Rural/farming table. | Local programs | N/A (no URL) |  |  |  |  |  |  |
| Animal | Aquatic monitoring | Rural/farming table. | AZ Game & Fish / ADEQ | N/A (no URL) |  |  |  |  |  |  |
| Animal | Animal sentinel program | GAP — "sentinels are needed"; no statewide animal database. | — | N/A (no URL) |  |  |  |  |  |  |
| Animal | Statewide rabies surveillance in animals | GAP — rabies data not collected into MEDSIS. | — | N/A (no URL) |  |  |  |  |  |  |
| Animal | Small-farm / backyard-farming animal data | GAP — not systematically tracked unless owners use a vet. | — | N/A (no URL) |  |  |  |  |  |  |
| Animal | Animal waste / AMR in runoff data | GAP — emphasized as needed for One Health integration. | — | N/A (no URL) |  |  |  |  |  |  |

## Environment (10 sources)

| Type | Source | Description (from notes) | URL | Status | Data Available | Machine Readable | Online | Update frequency | Data free to use | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Environment | [National Weather Service — HeatRisk](https://www.wpc.ncep.noaa.gov/heatrisk/) | NWS+CDC heat-health tier co-developed (the "tier system co-developed with NOAA & CDC" in Heat table). | [link](https://www.wpc.ncep.noaa.gov/heatrisk/) | Open | ✓ | ✓ | ✓ | ✓ | ✓ | NWS HeatRisk has Data Access tab with GeoTIFF/raster downloads; daily updates |
| Environment | [NOAA](https://www.noaa.gov/) | Parent agency for NWS; Tucson WFO partner. | [link](https://www.noaa.gov/) | Open | ✓ | ✓ | ✓ | ✓ | ✓ | NOAA provides NCEI/NCDC APIs and bulk downloads; constantly updated |
| Environment | [NIHHIS / Heat.gov](https://www.heat.gov/) | Federal heat-health information system (NOAA+CDC, 2015). | [link](https://www.heat.gov/) | Open | ✓ | ✓ | ✓ | ✓ | ✓ | Heat.gov has Data & Tools section with downloadable datasets |
| Environment | [Tree Equity Score](https://www.treeequityscore.org/) | Heat table — tree canopy vs. SVI/income/heat (Tucson adopted in 2021). | [link](https://www.treeequityscore.org/) | Open | ✓ | ✓ | ✓ | ✓ | ✓ | Tree Equity Score offers downloadable city scores (CSV/shapefile); free; updates with new ACS |
| Environment | Satellite imagery | Wildlife/VBD table. | e.g. Sentinel Hub, Landsat — generic | N/A (no URL) |  |  |  |  |  |  |
| Environment | [Water quality monitoring](https://www.azwater.gov/) | Heat table. | [link](https://www.azwater.gov/) / [link](https://azdeq.gov/) | Partial | ✓ | ✓ | ✓ | ✓ | ✓ | ADWR + ADEQ have GIS open-data hubs with downloads; URLs given are agency homepages |
| Environment | Land use data | Wildlife/VBD table. | State/county GIS | N/A (no URL) |  |  |  |  |  |  |
| Environment | [Environmental health data](https://www.azdhs.gov/preparedness/epidemiology-disease-control/environmental-health/) | Rural/tribal table. | [link](https://www.azdhs.gov/preparedness/epidemiology-disease-control/environmental-health/) | Partial | ✓ | ✗ | ✗ | ✓ | ✓ | AZ EPHT (Environmental Public Health Tracking) has dashboard; this URL is hub only |
| Environment | [ADHS Wastewater Surveillance](https://www.azdhs.gov/preparedness/epidemiology-disease-control/infectious-disease-epidemiology/index.php#wastewater-surveillance) | Info Flow — used with SVI for local vaccine work. | [link](https://www.azdhs.gov/preparedness/epidemiology-disease-control/infectious-disease-epidemiology/index.php#wastewater-surveillance) | Partial | ✓ | ✗ | ✗ | ✓ | ✓ | Wastewater surveillance is shown via dashboard, not bulk machine-readable feed |
| Environment | Energy use data | GAP — "single dataset most useful for identifying vulnerable energy users." Not released. | — | N/A (no URL) |  |  |  |  |  |  |

## Vector-borne (11 sources)

| Type | Source | Description (from notes) | URL | Status | Data Available | Machine Readable | Online | Update frequency | Data free to use | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Vector-borne | [ArboNET](https://www.cdc.gov/mosquitoes/php/arbonet/index.html) | CDC national arboviral surveillance — humans, blood donors, vets, mosquitoes, dead birds, sentinels. Named in Info Flow. | [link](https://www.cdc.gov/mosquitoes/php/arbonet/index.html) | N - inaccessible | ✗ | ✗ | ✗ | ✗ | ✗ | ArboNET URL 404; ArboNET data is available elsewhere via CDC but not at this URL |
| Vector-borne | [ADHS Mosquito-Borne Diseases program](https://www.azdhs.gov/preparedness/epidemiology-disease-control/mosquito-borne/index.php) | Trap-based; pools of 50 tested for WNV, encephalitis, dengue. | [link](https://www.azdhs.gov/preparedness/epidemiology-disease-control/mosquito-borne/index.php) | Info only | ✓ | ✗ | ✗ | ✓ | ✓ | Information page; disease data only via PDFs/dashboards |
| Vector-borne | [ADHS Vector-Borne & Zoonotic Diseases program](https://www.azdhs.gov/preparedness/epidemiology-disease-control/vector-borne-zoonotic-diseases/index.php) | Parent program — rabies, RMSF, mosquito-borne. | [link](https://www.azdhs.gov/preparedness/epidemiology-disease-control/vector-borne-zoonotic-diseases/index.php) | Info only | ✓ | ✗ | ✗ | ✓ | ✓ | Information page; data only via PDFs/dashboards |
| Vector-borne | [Great Arizona Tick Check](https://extension.arizona.edu/programs/great-arizona-tick-check) | Community-based participatory surveillance; UArizona + Cooperative Extension + ADHS, CDC-funded. Opportunistic at rabies clinics; ticks tested in tribal partner labs. | [link](https://extension.arizona.edu/programs/great-arizona-tick-check) | Closed | ✗ | ✗ | ✗ | ✗ | ✓ | Citizen-science program; database not publicly distributed |
| Vector-borne | [Great Arizona Mosquito Hunt (GAMH)](https://directorsblog.health.azdhs.gov/the-2015-great-arizona-mosquito-hunt/) | High-school citizen-science Aedes aegypti egg trapping, 2015–17 with ADHS + MCDPH + UArizona. | [link](https://directorsblog.health.azdhs.gov/the-2015-great-arizona-mosquito-hunt/) | Info only | ✗ | ✗ | ✗ | ✗ | ✓ | Blog post about 2015 program; no data |
| Vector-borne | Tribal mosquito surveillance | Cited as strongest existing tribal-led example. | Tribal-led, no public URL | N/A (no URL) |  |  |  |  |  |  |
| Vector-borne | Tribal partner labs (tick testing) | Where collected ticks are processed. | Tribal | N/A (no URL) |  |  |  |  |  |  |
| Vector-borne | Statewide tick surveillance | GAP — "no tick surveillance across most of Arizona"; largest geographic blind spot. | — | N/A (no URL) |  |  |  |  |  |  |
| Vector-borne | Comprehensive flea surveillance | GAP — "no comprehensive tick or flea surveillance." | — | N/A (no URL) |  |  |  |  |  |  |
| Vector-borne | Rodent surveillance | GAP — "no clear picture of rodent surveillance at all." | — | N/A (no URL) |  |  |  |  |  |  |
| Vector-borne | Pre-ER bite counts | GAP — wanted signal; most bites never reach the ER. | — | N/A (no URL) |  |  |  |  |  |  |

## Communication/Engagement (15 sources)

| Type | Source | Description (from notes) | URL | Status | Data Available | Machine Readable | Online | Update frequency | Data free to use | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Communication/Engagement | 24/7 hotlines | Complaints, exposure reports, general queries. | Agency-specific | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | 311 / citizen contact centers | Public complaint intake. | City-specific | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | Mass texting systems | "In development/consideration at some departments." | — | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | Event-based surveys | To assess how attendees heard about services. | Operational | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | Website / social-media analytics | "Where available." | Operational | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | [Health Alert Network (HAN) / AzHAN](https://han.health.azdhs.gov/) | Provider alerts (already listed under Human). | [link](https://han.health.azdhs.gov/) | Closed | ✗ | ✗ | ✗ | ✗ | ✗ | AzHAN — same as row 11; login-only |
| Communication/Engagement | Monthly provider calls | Cited under healthcare providers. | Operational | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | Welcome packets (new facilities) | Provider onboarding. | Operational | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | Radio (AM/FM, local, tribal-language) | Key for elders and remote communities. | Channel | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | PSAs / press releases / TV / print | Outbound. | Channel | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | Facebook, LinkedIn, YouTube | Wildlife table specifically — communications team uses Facebook, monitors clicks. | Channel | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | Word-of-mouth networks | Unhoused and tribal tables. | Informal | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | Flyers / handouts | Shelters, rec centers, gas stations, trading posts. | Channel | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | In-person events / health fairs | Cross-cutting. | Channel | N/A (no URL) |  |  |  |  |  |  |
| Communication/Engagement | Door-to-door qualitative data | Rural/tribal vaccination context — counts as both engagement and data collection. | Operational | N/A (no URL) |  |  |  |  |  |  |
