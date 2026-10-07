# Information Security Policy
**Document ID:** SEC-001 | **Version:** 2.4 | **Owner:** Chief Information Security Officer

## Purpose
This policy establishes the minimum-security controls that all Acme Corp employees, contractors, and systems must meet to protect confidential data, maintain regulatory compliance, and prevent unauthorised access.

## Scope
Applies to all staff, vendors with system access, and every device or service that processes Acme data — on-premises, cloud, or personal (BYOD).

## Password and Authentication Requirements
All user accounts must meet the following minimum standards:
- Passwords must be at least 16 characters and include uppercase letters, lowercase letters, digits, and special characters.
- Passwords must be rotated every 90 days; the last 10 passwords may not be reused.
- Multi-factor authentication (MFA) is mandatory for all corporate accounts including email, VPN, cloud consoles, and code repositories.
- Service accounts must use certificate-based authentication or secrets managed by the approved vault (HashiCorp Vault). Hardcoded credentials in source code are prohibited.
- Failed login attempts exceeding five consecutive failures trigger a 30-minute lockout and alert to the Security Operations Centre (SOC).

## Data Classification
Acme data is classified into four tiers:
1. **Public** – Marketing materials, published documentation. No controls beyond standard handling.
2. **Internal** – General business information. Accessible to all employees; not for external sharing without approval.
3. **Confidential** – Customer PII, financial records, HR data. Encrypted at rest and in transit; access on a need-to-know basis; audit logging required.
4. **Restricted** – Source code, cryptographic keys, M&A data. End-to-end encryption; access requires CISO approval; stored in approved isolated systems only.

## Access Control
- Access is provisioned on the principle of least privilege.
- Role assignments are reviewed quarterly by department heads and the Identity & Access Management (IAM) team.
- Termination of employment triggers immediate account deactivation, credential revocation, and hardware return within 24 hours.
- Privileged access (admin, root, DBA) requires a separate account, is logged in the privileged-access management (PAM) system, and must be justified via a change-management ticket.

## Incident Response
Security incidents follow the PICERL lifecycle:
1. **Preparation** – SOC maintains runbooks; tabletop exercises quarterly.
2. **Identification** – Any employee suspecting a breach must report immediately to security@acme.example via the #security-incidents Slack channel.
3. **Containment** – SOC isolates affected systems within one hour of confirmed incident.
4. **Eradication** – Root cause removed; patches applied; credentials rotated.
5. **Recovery** – Systems restored from clean backups; users notified; service validated.
6. **Lessons Learned** – Post-incident review within five business days; findings logged in the risk register.

## Security Training
- All employees complete mandatory security-awareness training during onboarding and annually thereafter.
- Phishing simulation campaigns run quarterly; employees who fail two consecutive simulations attend a remedial session.
- Engineers handling Restricted or Confidential data complete a 4-hour secure-coding course annually.

## Enforcement
Violations may result in disciplinary action up to and including termination, and referral to law enforcement where applicable. Employees who discover a policy violation must report it without fear of retaliation.

## Review Cycle
This policy is reviewed annually or after any significant security incident. The CISO approves all changes.
