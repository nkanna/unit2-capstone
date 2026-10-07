# Customer Complaint Handling Policy
**Document ID:** CS-003 | **Version:** 3.1 | **Owner:** VP of Customer Success

## Purpose
Every customer complaint is an opportunity to learn and retain trust. This policy ensures complaints are acknowledged quickly, resolved fairly, and used to drive product and process improvements.

## Scope
Applies to all complaints received through any channel — email, in-app support, phone, social media, or escalation from sales — about Acme products, services, billing, or staff conduct.

## Complaint Intake Channels
| Channel | Routing | Triage Owner |
|---|---|---|
| Support portal (support.acme.example) | Auto-ticket creation | Level-1 Support |
| Email (support@acme.example) | Parsed and auto-ticketed | Level-1 Support |
| Phone (+1-800-ACME-HELP) | Call log → ticket | Support Duty Manager |
| Social media (Twitter/X, LinkedIn) | Social monitoring tool → ticket | Community Manager |
| Sales escalation | Direct to Customer Success Manager | CS Manager |

All complaints must be assigned a unique ticket number within 30 minutes of receipt and an initial acknowledgement sent to the customer within the SLA window below.

## Severity Classification
Complaints are classified at intake:
- **P1 – Critical:** Service outage or data loss affecting the customer's core business operations. Estimated business impact > $10,000/day.
- **P2 – High:** Significant feature unavailability or billing error over $500.
- **P3 – Medium:** Non-critical feature degradation, incorrect documentation, or staff conduct complaint.
- **P4 – Low:** General feedback, cosmetic issues, or feature requests framed as complaints.

## Response SLAs
| Severity | Acknowledgement | First Response | Resolution Target |
|---|---|---|---|
| P1 | 15 minutes | 1 hour | 4 hours |
| P2 | 30 minutes | 4 hours | 1 business day |
| P3 | 1 hour | 1 business day | 3 business days |
| P4 | 4 hours | 3 business days | 10 business days |

Failure to meet P1 or P2 SLAs triggers an automatic page to the on-call Support Manager.

## Escalation Path
1. **Level-1 Support** handles P3 and P4 issues within their knowledge base.
2. **Level-2 Support (Tier Engineers)** handle technical issues requiring system access or code investigation.
3. **Customer Success Manager (CSM)** handles all P1/P2 issues, relationship-sensitive complaints, or any complaint where the customer explicitly requests manager contact.
4. **VP of Customer Success** handles complaints involving legal threats, regulatory inquiries, or potential churn of accounts > $50,000 ARR.
5. **Legal / Compliance** is looped in when complaints mention data breach, GDPR/CCPA, or contractual breach.

## Resolution and Documentation
- Every ticket must include root-cause notes before closing.
- Customer must confirm resolution or the ticket is auto-reopened after 48 hours.
- Resolutions involving a service credit or refund require CSM approval for amounts ≤ $1,000 and VP approval for amounts > $1,000.
- All P1/P2 resolutions are logged in the Incident Registry for trend analysis.

## Post-Resolution Follow-Up
Within five business days of P1/P2 resolution, the CSM must:
1. Send a personalised follow-up email confirming the fix and any preventive measures.
2. Offer a 30-minute call if the customer wishes to discuss further.
3. Log a Customer Health update in the CRM.

## Recurring Complaints and Trend Analysis
The VP of Customer Success reviews complaint metrics weekly:
- Volume by severity, channel, and product area.
- SLA attainment rates.
- Repeat complaints on the same issue (same customer or across customers).
Recurring themes are escalated to the Product and Engineering leads via the weekly Product-CS sync.
