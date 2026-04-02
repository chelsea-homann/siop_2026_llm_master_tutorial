# I-O Psychology Latent Variable Codebook

**for LLM-Based Persona Building in Disrupted Organizations**

SIOP 2026 Master Tutorial | Wymer, Wolfe, & Choe

---

## Purpose and Scope

This codebook defines 12 validated I-O psychology constructs that anchor every AI classification in the persona-building pipeline. Rather than allowing the AI to decide what themes are "important" on its own, the codebook provides a structured reference of constructs that have been validated in research and are relevant to organizational disruption.

### How This Codebook Functions in the Pipeline

The codebook serves three primary roles:

1. **Deductive classification (RAG Agent):** When the RAG Agent retrieves organizational documents, construct definitions and exemplars guide how retrieved content is matched to workforce themes. Each construct's operational definition serves as a semantic anchor for retrieval queries.

2. **Novel theme detection (Emergence Agent):** The Emergence Agent uses the codebook as a baseline. Any employee response pattern that does not map to one of these 12 constructs is flagged as a candidate emergent theme. The codebook thus defines the boundary between "expected" and "novel."

3. **Narrative grounding (Narrator Agent):** When the Narrator Agent writes persona descriptions, the codebook ensures that characterizations use validated construct language rather than ad hoc labels. This prevents the AI from inventing constructs that sound plausible but lack theoretical grounding.

### Relationship Between Survey Items and Codebook Constructs

The tutorial dataset includes five Likert-scale survey items: **Cared_About**, **Excited**, **Helpful_Info**, **Trust_Leaders**, and **Morale**. These five items are a simplified proxy designed for tutorial purposes. They do not directly measure any single construct in this codebook with the precision of a validated multi-item scale.

Instead, each survey item serves as a **partial indicator** that loads onto multiple constructs. For example:

- **Cared_About** partially reflects Perceived Organizational Support (POS) and Psychological Safety (PSY-SAF)
- **Excited** partially reflects Work Engagement (WRK-ENG) and Change Readiness (CHG-RDY)
- **Helpful_Info** partially reflects Communication Effectiveness (COMM-EFF) and Procedural Justice (JUST-PRO)
- **Trust_Leaders** partially reflects Trust in Leadership (TRUST-LDR) and Leader-Member Exchange (LMX)
- **Morale** partially reflects Organizational Commitment (ORG-COM) and Work-Life Balance (WLB)

In a full-scale implementation, each construct would be measured by its own validated multi-item scale (noted in each construct entry below). The five-item tutorial survey demonstrates the pipeline mechanics; the codebook demonstrates the construct precision that a production deployment would require.

---

## Construct Entries

---

### PSY-SAF: Psychological Safety

**Domain:** Team Climate

**Operational Definition:**
Psychological safety is a shared belief held by members of a team that the team is safe for interpersonal risk-taking. It describes a team climate characterized by interpersonal trust and mutual respect in which people are comfortable being themselves, raising concerns, asking questions, and admitting mistakes without fear of punishment or humiliation (Edmondson, 1999).

**Theoretical Grounding:**
Edmondson (1999) developed the construct through qualitative and quantitative research on learning behavior in work teams. She demonstrated that psychological safety predicts team learning behavior and performance beyond the effects of team structure, context support, and team coaching. The construct draws on earlier work on organizational climate (Schein & Bennis, 1965) and extends it to the team level with an emphasis on interpersonal risk.

**Survey Item Mapping:**
- **Primary proxy:** Cared_About (feeling valued by the organization reduces perceived interpersonal risk)
- **Secondary proxy:** Trust_Leaders (trust in leadership contributes to a climate where speaking up feels safe)
- **Full-scale measurement:** Edmondson's (1999) 7-item Psychological Safety Scale (e.g., "If you make a mistake on this team, it is often held against you" [reverse-scored]; "Members of this team are able to bring up problems and tough issues")

**Behavioral Exemplars During Disruption:**
- Team members openly voicing concerns about restructuring plans in meetings
- Employees asking clarifying questions about new role expectations without hesitation
- Managers sharing their own uncertainties about organizational changes to model vulnerability
- Team members flagging process problems during transitions without fear of retaliation

**Non-Examples:**
- *Organizational commitment* (PSY-SAF is about interpersonal risk on a team, not loyalty to the organization as a whole)
- *Trust in leadership* (PSY-SAF is a team-level climate property, not a dyadic assessment of a specific leader)
- *Job satisfaction* (an employee can feel satisfied with pay and benefits while still being afraid to speak up)

**Disruption Signals:**
During organizational disruption, psychological safety often deteriorates through *silence cascades*: one person withholds a concern, others notice and follow suit, and the team quickly shifts from open dialogue to self-protective silence. Restructurings amplify this effect because employees fear that raising concerns will mark them as resistant to change or make them targets for layoffs. Monitoring psychological safety during disruption is critical because its erosion is often invisible to leadership until it manifests as disengagement, turnover, or compliance without commitment.

---

### ORG-COM: Organizational Commitment

**Domain:** Attachment

**Operational Definition:**
Organizational commitment is a psychological state that characterizes the employee's relationship with the organization and has implications for the decision to continue or discontinue membership. Meyer and Allen (1991) identify three components: affective commitment (emotional attachment and identification), continuance commitment (perceived costs of leaving), and normative commitment (felt obligation to remain). In disruption contexts, the relative balance of these components determines whether retention reflects genuine engagement or economic constraint.

**Theoretical Grounding:**
Meyer and Allen (1991) proposed the three-component model of organizational commitment, which has become the dominant framework in I-O psychology. Meta-analytic evidence (Meyer, Stanley, Herscovitch, & Topolnytsky, 2002) confirms that affective commitment shows the strongest positive relationships with desirable outcomes (job performance, organizational citizenship behavior) and negative relationships with turnover, absenteeism, and counterproductive work behavior. Continuance commitment, by contrast, is unrelated or negatively related to performance.

**Survey Item Mapping:**
- **Primary proxy:** Morale (affective commitment and morale share variance through emotional identification with the organization)
- **Secondary proxy:** Cared_About (feeling cared about strengthens emotional attachment)
- **Full-scale measurement:** Meyer, Allen, and Smith's (1993) Revised Organizational Commitment Scale, with separate subscales for affective (6 items), continuance (6 items), and normative (6 items) commitment

**Behavioral Exemplars During Disruption:**
- Employees voluntarily taking on extra responsibilities during a transition period
- Long-tenure workers mentoring newer colleagues through organizational changes
- Employees advocating for the organization's mission to external stakeholders despite uncertainty
- Workers declining competing job offers because they believe in the organization's future

**Non-Examples:**
- *Job embeddedness* (a broader construct that includes community ties and off-the-job factors; Holtom, Mitchell, & Lee, 2006)
- *Work engagement* (engagement is about vigor and absorption in the work itself; commitment is about the bond with the organization)
- *Retention* (staying can reflect continuance commitment or lack of alternatives rather than genuine attachment)

**Disruption Signals:**
During organizational disruption, the three commitment components often shift in opposite directions. Affective commitment typically declines as employees question whether the organization still reflects their values. Continuance commitment may increase simultaneously if the job market tightens or severance packages create golden handcuffs. This creates a *retention risk shift*: the organization retains headcount but loses discretionary effort. Monitoring the balance between affective and continuance commitment during disruption reveals whether retention numbers mask an engagement crisis.

---

### POS: Perceived Organizational Support

**Domain:** Social Exchange

**Operational Definition:**
Perceived organizational support (POS) refers to employees' global beliefs concerning the extent to which the organization values their contributions and cares about their well-being. It reflects the employee's perception that the organization is committed to them, not just the reverse. POS operates through social exchange: employees who perceive support feel obligated to reciprocate through effort, loyalty, and organizational citizenship (Eisenberger, Huntington, Hutchison, & Sowa, 1986).

**Theoretical Grounding:**
Eisenberger et al. (1986) introduced POS within organizational support theory, grounded in social exchange theory (Blau, 1964) and the norm of reciprocity (Gouldner, 1960). Meta-analyses (Rhoades & Eisenberger, 2002) confirm that POS predicts job satisfaction, affective commitment, performance, and reduced turnover intentions. The construct is distinct from supervisor support (which feeds into POS) and organizational justice (which is an antecedent of POS).

**Survey Item Mapping:**
- **Primary proxy:** Cared_About (this item directly taps the "cares about well-being" component of POS)
- **Secondary proxy:** Helpful_Info (providing useful information signals organizational investment in employees)
- **Full-scale measurement:** Eisenberger, Huntington, Hutchison, and Sowa's (1986) Survey of Perceived Organizational Support (SPOS), typically shortened to 8 or 16 items (e.g., "The organization really cares about my well-being"; "The organization values my contribution to its well-being")

**Behavioral Exemplars During Disruption:**
- Organization providing additional mental health resources during restructuring
- Leaders holding transparent town halls about the impact of changes on employees
- Company maintaining development budgets for affected employees despite cost-cutting elsewhere
- HR proactively offering career transition support before layoffs are announced

**Non-Examples:**
- *Psychological safety* (PSY-SAF is about peer and team dynamics; POS is about the organization as an entity)
- *Supervisor support* (a related but distinct construct; supervisor support is one antecedent of POS, not POS itself)
- *Organizational commitment* (POS is the employee's perception of the organization's commitment to *them*; organizational commitment is the employee's commitment to the *organization*)

**Disruption Signals:**
POS functions as a *recovery predictor* during disruption. Organizations that maintained high POS before a disruption event see faster engagement recovery afterward, because the pre-existing social exchange relationship provides a buffer. When POS drops sharply during disruption, it signals that employees perceive the organization as violating the social exchange contract. This is particularly damaging because POS, once eroded, is slow to rebuild. Monitoring POS during disruption helps predict which workforce segments will recover and which will disengage permanently.

---

### CHG-RDY: Change Readiness

**Domain:** Change Management

**Operational Definition:**
Change readiness refers to the beliefs, attitudes, and intentions of organizational members regarding the extent to which changes are needed and the organization's capacity to successfully undertake those changes. It encompasses both cognitive appraisals (the change is necessary and feasible) and motivational states (willingness to engage in change-supportive behaviors). Armenakis, Harris, and Mossholder (1993) identify five key components of the change message that drive readiness: discrepancy (why change is needed), efficacy (confidence that change can succeed), appropriateness (the right change), principal support (leaders champion it), and valence (personal benefit).

**Theoretical Grounding:**
Armenakis et al. (1993) developed the readiness framework as a precursor to Lewin's unfreezing stage, arguing that readiness must be actively created through targeted communication. Holt, Armenakis, Feild, and Harris (2007) extended this by developing a multi-dimensional measure distinguishing individual readiness from organizational readiness. The construct bridges change management theory (Kotter, 1996) and social cognitive theory (Bandura, 1986) by emphasizing that employees' confidence in both the organization's and their own capacity for change determines engagement.

**Survey Item Mapping:**
- **Primary proxy:** Excited (excitement about the future reflects positive valence toward upcoming changes)
- **Secondary proxy:** Morale (sustained morale during transition signals that employees perceive the change as manageable)
- **Full-scale measurement:** Holt, Armenakis, Feild, and Harris's (2007) Readiness for Organizational Change Scale, a 25-item measure with four dimensions: appropriateness, management support, change efficacy, and personal valence

**Behavioral Exemplars During Disruption:**
- Employees proactively learning new tools or processes required by restructuring
- Team leads volunteering to pilot new workflows before formal rollout
- Workers asking constructive questions about implementation timelines rather than resisting
- Employees helping colleagues adapt to new reporting structures

**Non-Examples:**
- *Change resistance* (the absence of readiness is not simply resistance; Piderit, 2000, argues that ambivalence is more common than outright opposition)
- *Adaptability* (a dispositional trait; readiness is a state that can be influenced by organizational action)
- *Organizational commitment* (an employee can be committed to the organization but not ready for a specific change initiative)

**Disruption Signals:**
Change readiness functions as a *transformation gate*: organizational changes succeed or fail at this threshold. Low readiness does not mean employees are hostile to change; it typically means the five message components (discrepancy, efficacy, appropriateness, principal support, valence) have not been adequately communicated. During disruption, readiness is unevenly distributed across the workforce. Frontline employees who see disruption directly may have high discrepancy awareness but low efficacy. Senior leaders may have high efficacy but underestimate discrepancy at lower levels. Persona-based analysis can surface these asymmetries.

---

### ROL-AMB: Role Ambiguity

**Domain:** Role Stress

**Operational Definition:**
Role ambiguity exists when an employee lacks clear information about work role expectations, methods for fulfilling known role expectations, or the consequences of role performance. It is the degree to which required information is unavailable to a given organizational position (Rizzo, House, & Lirtzman, 1970). Unlike role conflict (simultaneous incompatible demands), role ambiguity is specifically about uncertainty in what is expected.

**Theoretical Grounding:**
Rizzo et al. (1970) developed role ambiguity within role theory, drawing on Kahn, Wolfe, Quinn, Snoek, and Rosenthal's (1964) seminal work on organizational stress. Meta-analyses (Tubre & Collins, 2000; Gilboa, Shirom, Fried, & Cooper, 2008) confirm that role ambiguity negatively predicts job performance and positively predicts anxiety, emotional exhaustion, and turnover intentions. The effect is particularly strong when employees cannot obtain clarification through informal channels, which is common during restructurings that disrupt established communication networks.

**Survey Item Mapping:**
- **Primary proxy:** Helpful_Info (receiving helpful information about one's role reduces ambiguity; low scores suggest information gaps)
- **Secondary proxy:** Trust_Leaders (employees who trust leadership are more likely to seek and receive role clarification)
- **Full-scale measurement:** Rizzo, House, and Lirtzman's (1970) 6-item Role Ambiguity Scale (e.g., "I know exactly what is expected of me" [reverse-scored]; "I know what my responsibilities are" [reverse-scored]). Note: items are typically reverse-scored so that higher scores indicate greater ambiguity.

**Behavioral Exemplars During Disruption:**
- Employees asking multiple managers who their new supervisor is after a reorganization
- Workers duplicating tasks because no one clarified which team now owns a process
- New hires receiving conflicting onboarding instructions from different departments
- Employees defaulting to old procedures because new ones have not been communicated

**Non-Examples:**
- *Role conflict* (receiving clear but contradictory demands from two supervisors is conflict, not ambiguity)
- *Role overload* (knowing exactly what is expected but having too much of it)
- *Job insecurity* (fear of losing one's job, which is a separate stressor from not knowing what the job requires)

**Disruption Signals:**
Restructurings are a primary driver of role ambiguity because they sever established *reporting line* relationships. Employees lose clarity on three fronts simultaneously: who they report to, what they are responsible for, and how their performance will be evaluated. This creates *reporting line confusion* that can persist for months after the formal reorganization is announced. In persona analysis, clusters with high role ambiguity during disruption often overlap with low communication effectiveness and low trust in leadership, suggesting that ambiguity is a downstream consequence of inadequate change communication.

---

### LMX: Leader-Member Exchange

**Domain:** Leadership

**Operational Definition:**
Leader-member exchange describes the quality of the dyadic relationship between a supervisor and a subordinate. High-LMX relationships are characterized by mutual trust, respect, and obligation that go beyond the formal employment contract. Low-LMX relationships are limited to the transactional terms of the job description. The construct emphasizes that leaders do not treat all subordinates uniformly; instead, they develop differentiated relationships that vary in quality (Graen & Uhl-Bien, 1995).

**Theoretical Grounding:**
Graen and Uhl-Bien (1995) codified LMX theory, which evolved from Vertical Dyad Linkage theory (Dansereau, Graen, & Haga, 1975). Meta-analytic evidence (Gerstner & Day, 1997; Martin, Guillaume, Thomas, Lee, & Epitropaki, 2016) demonstrates that LMX quality predicts job satisfaction, organizational commitment, job performance, and turnover intentions. The dyadic nature of LMX distinguishes it from general leadership climate measures: two employees in the same team can have very different LMX relationships with the same leader.

**Survey Item Mapping:**
- **Primary proxy:** Trust_Leaders (trust is a core component of high-quality LMX)
- **Secondary proxy:** Cared_About (feeling cared about by one's direct leader reflects the affective dimension of LMX)
- **Full-scale measurement:** Graen and Uhl-Bien's (1995) LMX-7 Scale (e.g., "How well does your leader understand your job problems and needs?"; "How would you characterize your working relationship with your leader?")

**Behavioral Exemplars During Disruption:**
- A manager advocating for their direct report's role during restructuring decisions
- An employee receiving advance information about organizational changes from their supervisor
- A leader providing emotional support and career guidance to team members facing uncertainty
- An employee and their manager collaboratively developing a transition plan for the new structure

**Non-Examples:**
- *Transformational leadership* (a leadership style; LMX is about relationship quality regardless of the leader's style)
- *Trust in leadership* (TRUST-LDR is about the broader leadership team; LMX is specifically dyadic)
- *Psychological safety* (a team-level climate; LMX is a relationship between two individuals)

**Disruption Signals:**
LMX is acutely sensitive to *manager reassignment* during restructuring. When employees are moved to new supervisors, high-quality LMX relationships built over months or years are severed. The new relationship starts at the transactional baseline, even if the employee previously enjoyed a high-quality partnership. This creates a double burden: the employee deals with organizational change and simultaneously loses a key source of support. In persona analysis, clusters with high pre-disruption LMX that experience manager turnover may show steeper declines in engagement and commitment than clusters that maintained supervisor continuity.

---

### JUST-PRO: Procedural Justice

**Domain:** Justice

**Operational Definition:**
Procedural justice refers to the perceived fairness of the processes and procedures used to make decisions that affect employees. It is distinct from distributive justice (fairness of outcomes) and interactional justice (fairness of interpersonal treatment). Procedures are perceived as fair when they are consistent, unbiased, based on accurate information, correctable, representative of all parties' concerns, and aligned with prevailing ethical standards (Leventhal, 1980; Colquitt, 2001).

**Theoretical Grounding:**
Colquitt (2001) validated a four-dimensional model of organizational justice, demonstrating that procedural, distributive, interpersonal, and informational justice are empirically distinct. Procedural justice is particularly important during organizational change because it predicts acceptance of unfavorable outcomes: employees who perceive the decision-making process as fair are more likely to accept even negative results (Brockner & Wiesenfeld, 1996). This "fair process effect" makes procedural justice a critical lever for change management.

**Survey Item Mapping:**
- **Primary proxy:** Helpful_Info (access to transparent, accurate information is a core procedural justice criterion)
- **Secondary proxy:** Trust_Leaders (procedural justice judgments drive trust; employees who perceive fair processes trust the decision-makers)
- **Full-scale measurement:** Colquitt's (2001) Procedural Justice Scale, a 7-item measure (e.g., "Have you been able to express your views and feelings during those procedures?"; "Have those procedures been applied consistently?")

**Behavioral Exemplars During Disruption:**
- Organization publishing clear criteria for how restructuring decisions will be made
- HR providing a formal appeals process for role reassignment decisions
- Leaders explaining the data and reasoning behind each restructuring choice
- Employees receiving consistent information about selection criteria across departments

**Non-Examples:**
- *Distributive justice* (whether the outcome itself is fair, e.g., whether severance packages are equitable)
- *Interactional justice* (whether managers treat employees with dignity and respect when delivering decisions)
- *Communication effectiveness* (information can be effectively communicated but still describe an unfair process)

**Disruption Signals:**
Procedural justice is the primary driver of *process fairness* perceptions during organizational change. When restructuring decisions appear arbitrary, inconsistent, or opaque, procedural justice perceptions collapse. This matters because procedural justice violations produce longer-lasting damage than distributive justice violations: an employee can accept a bad outcome from a fair process, but a good outcome from an unfair process still erodes trust. In persona analysis, clusters with low procedural justice during disruption often show a characteristic pattern of high cynicism, low commitment, and active disengagement, even when the objective outcomes of the change are neutral or positive for that cluster.

---

### TRUST-LDR: Trust in Leadership

**Domain:** Governance

**Operational Definition:**
Trust in leadership is the willingness of a party to be vulnerable to the actions of the leadership of the organization, based on the expectation that leadership will perform actions important to the trustor, irrespective of the ability to monitor or control the leadership (adapted from Mayer, Davis, & Schoorman, 1995). It encompasses assessments of leaders' ability (competence to enact their roles), benevolence (concern for employees' interests), and integrity (adherence to principles acceptable to employees).

**Theoretical Grounding:**
Mayer et al. (1995) proposed the integrative model of organizational trust, identifying ability, benevolence, and integrity as the three antecedents of trust. Their model distinguishes trust (a willingness to be vulnerable) from trustworthiness (the characteristics that make one worthy of trust). Dirks and Ferrin (2002) meta-analytically confirmed that trust in leadership predicts job satisfaction, organizational commitment, and organizational citizenship behavior, with trust in direct leaders showing stronger effects on performance and trust in organizational leaders showing stronger effects on commitment.

**Survey Item Mapping:**
- **Primary proxy:** Trust_Leaders (this item directly taps trust in leadership)
- **Secondary proxy:** Morale (trust in leadership sustains morale during uncertainty; their erosion tends to co-occur)
- **Full-scale measurement:** Mayer and Davis's (1999) Organizational Trust Inventory, with separate subscales for ability, benevolence, and integrity; or the short form by Schoorman, Mayer, and Davis (2007)

**Behavioral Exemplars During Disruption:**
- Employees accepting temporary role ambiguity because they trust that leaders will clarify soon
- Workers sharing candid feedback in skip-level meetings without fear of retaliation
- Employees giving leadership the benefit of the doubt when early change communications are incomplete
- Teams continuing to invest discretionary effort despite uncertainty about the organization's direction

**Non-Examples:**
- *Leader-member exchange* (LMX is about the quality of a specific supervisor-subordinate relationship; TRUST-LDR is about the broader leadership team or senior leadership)
- *Psychological safety* (PSY-SAF is a team-level climate; TRUST-LDR is a vertical assessment of the leadership hierarchy)
- *Procedural justice* (a related antecedent of trust; perceiving fair processes builds trust, but they are empirically distinct constructs)

**Disruption Signals:**
Trust in leadership is subject to *credibility erosion* during disruption, and the trajectory is asymmetric: trust declines rapidly but rebuilds slowly. A single instance of perceived deception or broken commitment can undo months of trust-building. During restructuring, the most common trust violations involve information asymmetry (leaders knew about changes before employees did), inconsistency (different messages to different groups), and perceived self-interest (leadership appears to protect itself at employees' expense). In persona analysis, clusters with eroding trust often show a leading indicator pattern: trust declines before engagement, commitment, and morale follow.

---

### WRK-ENG: Work Engagement

**Domain:** Motivation

**Operational Definition:**
Work engagement is a positive, fulfilling, work-related state of mind characterized by vigor (high levels of energy and mental resilience while working), dedication (sense of significance, enthusiasm, and challenge), and absorption (full concentration and happy engrossment in work). It represents the opposite of burnout along two dimensions: vigor is the opposite of exhaustion, and dedication is the opposite of cynicism (Schaufeli, Salanova, Gonzalez-Roma, & Bakker, 2002).

**Theoretical Grounding:**
Schaufeli et al. (2002) developed the Utrecht Work Engagement Scale (UWES) within the Job Demands-Resources (JD-R) model (Bakker & Demerouti, 2007). The JD-R model posits that job resources (autonomy, feedback, social support) drive engagement through a motivational process, while job demands (workload, time pressure, emotional demands) drive burnout through a health impairment process. Meta-analytic evidence (Christian, Garza, & Slaughter, 2011) confirms that engagement predicts task performance, contextual performance, and reduced turnover beyond the effects of job satisfaction and organizational commitment.

**Survey Item Mapping:**
- **Primary proxy:** Excited (excitement maps to the dedication dimension of engagement: enthusiasm and challenge)
- **Secondary proxy:** Morale (sustained morale reflects the vigor dimension: energy and resilience)
- **Full-scale measurement:** Schaufeli et al.'s (2002) Utrecht Work Engagement Scale (UWES-17 or UWES-9), measuring vigor (e.g., "At my work, I feel bursting with energy"), dedication (e.g., "I am enthusiastic about my job"), and absorption (e.g., "I am immersed in my work")

**Behavioral Exemplars During Disruption:**
- Employees voluntarily staying late to complete transition-related tasks
- Workers generating creative solutions to problems caused by restructuring
- Team members mentoring colleagues through new processes with genuine enthusiasm
- Employees maintaining high-quality output despite organizational uncertainty

**Non-Examples:**
- *Workaholism* (working excessively from compulsion, not from positive engagement; workaholism correlates with poor well-being while engagement correlates with positive well-being)
- *Job satisfaction* (a cognitive evaluation of the job; engagement is an active, motivational state)
- *Organizational commitment* (commitment is about the bond with the organization; engagement is about the relationship with the work itself)

**Disruption Signals:**
Work engagement is a *downstream outcome* that reflects the combined impact of multiple disruption-related changes. It typically does not drop first; instead, it follows declines in trust, POS, and change readiness. This makes engagement a lagging indicator: by the time engagement scores decline significantly, the root causes have been accumulating for weeks or months. In persona analysis, clusters that maintain high engagement during disruption usually also show high POS and trust, suggesting that engagement resilience is not an individual trait but a function of organizational context. Clusters with suddenly plummeting engagement warrant investigation of upstream drivers rather than direct engagement interventions.

---

### COMM-EFF: Communication Effectiveness

**Domain:** Sensemaking

**Operational Definition:**
Communication effectiveness during organizational change refers to the extent to which organizational communications reduce uncertainty, are perceived as timely, accurate, complete, and useful, and enable employees to make sense of ambiguous situations. It goes beyond message delivery to include message reception, comprehension, and the degree to which communication enables employees to construct coherent narratives about organizational events (Bordia, Hobman, Jones, Gallois, & Callan, 2004).

**Theoretical Grounding:**
Bordia et al. (2004) demonstrated that quality of change communication predicts psychological strain during uncertainty, mediated by perceptions of control. Their work builds on Weick's (1995) sensemaking framework: during disruption, employees actively construct meaning from available information, and the quality of organizational communication determines whether that meaning-making produces adaptive or maladaptive responses. DiFonzo and Bordia (1998) further showed that communication voids are filled by rumors, which amplify anxiety and erode trust.

**Survey Item Mapping:**
- **Primary proxy:** Helpful_Info (receiving helpful information is the direct behavioral indicator of effective communication)
- **Secondary proxy:** Trust_Leaders (effective communication builds trust; its absence erodes trust through information vacuums)
- **Full-scale measurement:** Bordia et al.'s (2004) change communication quality measures, or Miller, Johnson, and Grau's (1994) 5-item communication satisfaction subscale adapted for change contexts

**Behavioral Exemplars During Disruption:**
- Organization sending timely updates about restructuring milestones as they occur
- Leaders holding open Q&A sessions where difficult questions receive honest answers
- HR providing clear, role-specific guides explaining how the reorganization affects each department
- Managers translating corporate announcements into team-specific implications

**Non-Examples:**
- *Communication frequency* (more communication is not necessarily more effective; poorly crafted frequent messages can increase confusion)
- *Transparency* (a related value, but effectiveness includes clarity, relevance, and actionability beyond mere openness)
- *Information sharing* (a unidirectional concept; communication effectiveness includes reception and comprehension)

**Disruption Signals:**
Communication effectiveness is the primary buffer against the *information vacuum* that naturally forms during organizational disruption. When official channels go silent or provide vague, corporate-speak responses, employees fill the void with rumors, worst-case speculation, and peer-to-peer interpretation. The result is fragmented sensemaking: different workforce segments construct entirely different narratives about the same organizational event. In persona analysis, clusters with low communication effectiveness scores during disruption often show high variability on other constructs, because the absence of authoritative information allows attitudes to drift in idiosyncratic directions.

---

### CAR-DEV: Career Development

**Domain:** Growth

**Operational Definition:**
Career development refers to employees' perceptions of the organization's investment in their professional growth, skill acquisition, and career advancement opportunities. It encompasses both formal programs (training, mentoring, succession planning) and informal signals (manager conversations about career goals, visibility assignments, stretch opportunities). The construct captures the perceived availability and quality of developmental support, not just its existence on paper (Kraimer, Seibert, Wayne, Liden, & Bravo, 2011).

**Theoretical Grounding:**
Kraimer et al. (2011) demonstrated that perceived organizational support for development predicts career satisfaction and organizational commitment beyond the effects of general POS. Their work builds on career construction theory (Savickas, 2005), which positions career development as a joint enterprise between the individual and the organization. In the JD-R model (Bakker & Demerouti, 2007), developmental resources function as a key job resource that drives engagement through the motivational pathway.

**Survey Item Mapping:**
- **Primary proxy:** Excited (excitement about one's future at the organization reflects perceived development opportunities)
- **Secondary proxy:** Helpful_Info (receiving useful career-relevant information signals organizational investment in growth)
- **Full-scale measurement:** Kraimer et al.'s (2011) Perceived Organizational Support for Development Scale (e.g., "My organization provides me with opportunities to develop my career"); or Turban and Dougherty's (1994) career mentoring measures

**Behavioral Exemplars During Disruption:**
- Organization maintaining professional development budgets during restructuring rather than cutting them first
- Managers proactively discussing career paths in the new organizational structure with their reports
- Company offering reskilling programs for employees whose roles are changing
- HR providing internal mobility tools that show available positions in the new structure

**Non-Examples:**
- *Perceived organizational support* (POS is broader, covering well-being and contribution valuation; CAR-DEV is specifically about growth and advancement)
- *Job satisfaction* (an employee can be satisfied with their current role while perceiving no development opportunities)
- *Self-directed learning* (an individual behavior; CAR-DEV is about organizational provision, not personal initiative)

**Disruption Signals:**
During disruption, career development perceptions shift from abstract to urgent as employees reassess their *mobility perception*: can I advance here, or do I need to advance elsewhere? Restructurings simultaneously create development opportunities (new roles, expanded responsibilities) and threats (eliminated positions, frozen promotions). The persona analysis can reveal which workforce segments perceive the disruption as a career catalyst versus a career threat. Clusters with high pre-disruption CAR-DEV that experience sharp declines may represent talent at highest flight risk, because these employees previously expected growth and now perceive stagnation.

---

### WLB: Work-Life Balance

**Domain:** Well-Being

**Operational Definition:**
Work-life balance refers to the degree to which an individual's effectiveness and satisfaction in work and family roles are compatible with the individual's life priorities. It encompasses the absence of inter-role conflict (where demands of one role make it difficult to fulfill another) and the presence of inter-role enrichment (where experiences in one role improve quality of life in another). The construct is subjective: what constitutes "balance" depends on the individual's values and life stage (Greenhaus & Beutell, 1985).

**Theoretical Grounding:**
Greenhaus and Beutell (1985) identified three forms of work-family conflict: time-based (time devoted to one role makes it difficult to fulfill another), strain-based (strain produced by one role spills over into another), and behavior-based (specific behaviors required in one role are incompatible with expectations in another). Subsequent work by Greenhaus and Allen (2011) expanded the framework to include the positive side: work-family enrichment, where resources gained in one domain transfer to improve functioning in the other.

**Survey Item Mapping:**
- **Primary proxy:** Morale (sustained morale often reflects sustainable work-life integration; morale erosion frequently signals imbalance)
- **Secondary proxy:** Cared_About (feeling cared about includes perceptions that the organization respects boundaries and personal needs)
- **Full-scale measurement:** Carlson, Kacmar, and Williams's (2000) Work-Family Conflict Scale (18 items across six dimensions: time, strain, and behavior-based conflict in both work-to-family and family-to-work directions)

**Behavioral Exemplars During Disruption:**
- Employees reporting increased after-hours work demands due to restructuring deadlines
- Workers struggling to maintain childcare routines when in-office requirements change
- Team members covering multiple roles during a transition and losing personal time
- Employees expressing concern that the new organizational structure will require more travel

**Non-Examples:**
- *Job satisfaction* (an employee can be satisfied with their job content while experiencing severe work-life conflict)
- *Work engagement* (high absorption in work can coexist with or even contribute to poor work-life balance)
- *Burnout* (a consequence of sustained imbalance, not the imbalance itself)

**Disruption Signals:**
Organizational disruption intensifies work-life conflict through *workload redistribution*: when teams shrink or merge, surviving employees absorb additional responsibilities. This time-based and strain-based conflict is compounded when restructurings change work modalities (e.g., shifting from remote to hybrid). The effects are not uniformly distributed; employees with caregiving responsibilities, long commutes, or dual-career partnerships experience disproportionate impact. In persona analysis, clusters with declining WLB scores during disruption are critical to identify because work-life imbalance is a leading predictor of both health outcomes and voluntary turnover.

---

## References

- Armenakis, A. A., Harris, S. G., & Mossholder, K. W. (1993). Creating readiness for organizational change. *Human Relations, 46*(6), 681-703.
- Bakker, A. B., & Demerouti, E. (2007). The Job Demands-Resources model: State of the art. *Journal of Managerial Psychology, 22*(3), 309-328.
- Bandura, A. (1986). *Social foundations of thought and action: A social cognitive theory.* Prentice-Hall.
- Blau, P. M. (1964). *Exchange and power in social life.* Wiley.
- Bordia, P., Hobman, E., Jones, E., Gallois, C., & Callan, V. J. (2004). Uncertainty during organizational change. *Journal of Business and Psychology, 18*(4), 507-532.
- Brockner, J., & Wiesenfeld, B. M. (1996). An integrative framework for explaining reactions to decisions. *Psychological Bulletin, 120*(2), 189-208.
- Carlson, D. S., Kacmar, K. M., & Williams, L. J. (2000). Construction and initial validation of a multidimensional measure of work-family conflict. *Journal of Vocational Behavior, 56*(2), 249-276.
- Christian, M. S., Garza, A. S., & Slaughter, J. E. (2011). Work engagement: A quantitative review and test of its relations with task and contextual performance. *Personnel Psychology, 64*(1), 89-136.
- Colquitt, J. A. (2001). On the dimensionality of organizational justice. *Journal of Applied Psychology, 86*(3), 386-400.
- Dansereau, F., Graen, G., & Haga, W. J. (1975). A vertical dyad linkage approach to leadership within formal organizations. *Organizational Behavior and Human Performance, 13*(1), 46-78.
- DiFonzo, N., & Bordia, P. (1998). A tale of two corporations: Managing uncertainty during organizational change. *Human Resource Management, 37*(3-4), 295-303.
- Dirks, K. T., & Ferrin, D. L. (2002). Trust in leadership: Meta-analytic findings and implications for research and practice. *Journal of Applied Psychology, 87*(4), 611-628.
- Edmondson, A. (1999). Psychological safety and learning behavior in work teams. *Administrative Science Quarterly, 44*(2), 350-383.
- Eisenberger, R., Huntington, R., Hutchison, S., & Sowa, D. (1986). Perceived organizational support. *Journal of Applied Psychology, 71*(3), 500-507.
- Gerstner, C. R., & Day, D. V. (1997). Meta-analytic review of leader-member exchange theory. *Journal of Applied Psychology, 82*(6), 827-844.
- Gilboa, S., Shirom, A., Fried, Y., & Cooper, C. (2008). A meta-analysis of work demand stressors and job performance. *Personnel Psychology, 61*(2), 227-271.
- Gouldner, A. W. (1960). The norm of reciprocity. *American Sociological Review, 25*(2), 161-178.
- Graen, G. B., & Uhl-Bien, M. (1995). Relationship-based approach to leadership. *The Leadership Quarterly, 6*(2), 219-247.
- Greenhaus, J. H., & Allen, T. D. (2011). Work-family balance: A review and extension of the literature. In J. C. Quick & L. E. Tetrick (Eds.), *Handbook of occupational health psychology* (2nd ed., pp. 165-183). American Psychological Association.
- Greenhaus, J. H., & Beutell, N. J. (1985). Sources of conflict between work and family roles. *Academy of Management Review, 10*(1), 76-88.
- Holt, D. T., Armenakis, A. A., Feild, H. S., & Harris, S. G. (2007). Readiness for organizational change: The systematic development of a scale. *The Journal of Applied Behavioral Science, 43*(2), 232-255.
- Holtom, B. C., Mitchell, T. R., & Lee, T. W. (2006). Increasing human and social capital by applying job embeddedness theory. *Organizational Dynamics, 35*(4), 316-331.
- Kahn, R. L., Wolfe, D. M., Quinn, R. P., Snoek, J. D., & Rosenthal, R. A. (1964). *Organizational stress: Studies in role conflict and ambiguity.* Wiley.
- Kotter, J. P. (1996). *Leading change.* Harvard Business School Press.
- Kraimer, M. L., Seibert, S. E., Wayne, S. J., Liden, R. C., & Bravo, J. (2011). Antecedents and outcomes of organizational support for development. *Journal of Applied Psychology, 96*(3), 485-500.
- Leventhal, G. S. (1980). What should be done with equity theory? In K. J. Gergen, M. S. Greenberg, & R. H. Willis (Eds.), *Social exchange: Advances in theory and research* (pp. 27-55). Plenum.
- Martin, R., Guillaume, Y., Thomas, G., Lee, A., & Epitropaki, O. (2016). Leader-member exchange (LMX) and performance: A meta-analytic review. *Personnel Psychology, 69*(1), 67-121.
- Mayer, R. C., & Davis, J. H. (1999). The effect of the performance appraisal system on trust for management. *Journal of Applied Psychology, 84*(1), 123-136.
- Mayer, R. C., Davis, J. H., & Schoorman, F. D. (1995). An integrative model of organizational trust. *Academy of Management Review, 20*(3), 709-734.
- Meyer, J. P., & Allen, N. J. (1991). A three-component conceptualization of organizational commitment. *Human Resource Management Review, 1*(1), 61-89.
- Meyer, J. P., Allen, N. J., & Smith, C. A. (1993). Commitment to organizations and occupations: Extension and test of a three-component conceptualization. *Journal of Applied Psychology, 78*(4), 538-551.
- Meyer, J. P., Stanley, D. J., Herscovitch, L., & Topolnytsky, L. (2002). Affective, continuance, and normative commitment to the organization: A meta-analysis. *Journal of Vocational Behavior, 61*(1), 20-52.
- Miller, V. D., Johnson, J. R., & Grau, J. (1994). Antecedents to willingness to participate in a planned organizational change. *Journal of Applied Communication Research, 22*(1), 59-80.
- Piderit, S. K. (2000). Rethinking resistance and recognizing ambivalence: A multidimensional view of attitudes toward an organizational change. *Academy of Management Review, 25*(4), 783-794.
- Rhoades, L., & Eisenberger, R. (2002). Perceived organizational support: A review of the literature. *Journal of Applied Psychology, 87*(4), 698-714.
- Rizzo, J. R., House, R. J., & Lirtzman, S. I. (1970). Role conflict and ambiguity in complex organizations. *Administrative Science Quarterly, 15*(2), 150-163.
- Savickas, M. L. (2005). The theory and practice of career construction. In S. D. Brown & R. W. Lent (Eds.), *Career development and counseling* (pp. 42-70). Wiley.
- Schaufeli, W. B., Salanova, M., Gonzalez-Roma, V., & Bakker, A. B. (2002). The measurement of engagement and burnout. *Journal of Happiness Studies, 3*(1), 71-92.
- Schein, E. H., & Bennis, W. G. (1965). *Personal and organizational change through group methods.* Wiley.
- Schoorman, F. D., Mayer, R. C., & Davis, J. H. (2007). An integrative model of organizational trust: Past, present, and future. *Academy of Management Review, 32*(2), 344-354.
- Tubre, T. C., & Collins, J. M. (2000). Jackson and Schuler (1985) revisited: A meta-analysis of the relationships between role ambiguity, role conflict, and job performance. *Journal of Management, 26*(1), 155-169.
- Turban, D. B., & Dougherty, T. W. (1994). Role of protege personality in receipt of mentoring and career success. *Academy of Management Journal, 37*(3), 688-702.
- Weick, K. E. (1995). *Sensemaking in organizations.* SAGE Publications.
