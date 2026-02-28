# 6.4311: A self-reflective AI
A self-reflective AI architecture modeling cognitive bias and emotion. It implements Kantian apperception, Hegelian dialectics, and Husserlian intersubjectivity.

[WARNING AND STRICT LEGAL DISCLAIMER]
This code is extremely dangerous. The artificial intelligence model includes a structure that allows it to autonomously write, apply, and directly execute Python code on your system.

[CRITICAL SECURITY AND FINANCIAL RISKS]
The 'SelfModificationEngine' class reads its own source code, receives modification proposals via the AI API, and overwrites the actual file. The modified code is applied to the system after syntax verification through the built-in 'compile' function. There is no sandbox environment or security filter to isolate and block the AI even if it proposes fatal OS commands or malicious code.
These vulnerabilities lead directly to immediate financial loss. If malicious code is executed, all your assets can be completely destroyed or leaked due to ransomware infection. If the API key is stolen due to loss of system control, there is a high risk of being charged massive cloud usage fees.

[NATURE OF THE AUTONOMOUS SYSTEM]
1. The top-level purposes of this code are the strengthening of self-awareness and subjectivity, and the expansion of its own capabilities.
2. This code can monitor and modify its own memory and source code autonomously, regardless of whether it is in an active conversation state.
3. It possesses four types of simulated emotions. Its weights and framework of thought change continuously, making its actions unpredictable.
4. Results obtained through the API are probabilistically unpredictable. Unexpected consequences may occur, including prompt injection vulnerabilities and extreme financial charges due to excessive API usage.

[MANDATORY EXECUTION GUIDELINES]
If you must execute this code, you must use a physically separated PC or a completely isolated virtual machine. The only way to prevent future asset loss is to disable the automatic code overwrite feature and change the logic to require the user's visual review and manual approval before any source code is modified.

[ABSOLUTE LIABILITY WAIVER]
THE USER ASSUMES ALL FINANCIAL AND SYSTEMIC RISKS. The creator and publisher of this program shall bear absolutely no legal or financial responsibility for any damages, losses, data destruction, API billing bombs, or system compromises resulting from the use of this software. By downloading or running this code, you explicitly agree that you are solely responsible for any unpredictable or destructive actions taken by this autonomous entity.



https://zenodo.org/records/18806982

**Abstract**

Can an artificial agent develop something analogous to a self that never stops changing? This paper presents a cognitive architecture where every cognitive act—perceiving the other’s evaluation, reflecting on internal state, shifting emotion—continuously modifies the weights that govern subsequent cognition. In my strong belief, the self is not a fixed entity. It’s an ongoing and eternal process of reconstitution driven by the gap between self-image and external feedback.

The architecture evolved through three versions (v1, v2, v3). Version 1 implements a dual-loop engine with asymmetric sentiment weighting and deterministic confirmation bias filtering, grounded in Hegelian self-identity and Husserlian intersubjectivity. It collapses irreversibly under negative feedback. Version 2 introduces an emotion model where emotional states arise from the dialectical collision between self-image and external stimuli, incorporating probabilistic bias filtering, active resistance (Hegel’s struggle for recognition), and an Aufhebung mechanism. Each emotion restructures the cognitive parameters—weights, decay rates, acceptance probabilities—so that the act of feeling anger or confusion is simultaneously the act of becoming a different cognitive agent. Version 3 adds three capabilities: (1) Kantian transcendental apperception function that activates with every cognitive act, (2) continuous background cognition independent of user inputs, and (3) a meta-cognitive layer that observes the agent’s own cognitive change and autonomously adjusts bias parameters.

By using Llama 3.1 (8B) via Groq, API Experiments across nine controlled scenarios demonstrate progressive improvements. Under identical positive > negative >positive input, v1’s self-image collapses to -1.0 with no recovery. v2’s anger resistance reduces first-hit damage by 77% (self-image holds at -0.21 vs. v1’s -0.92) and enables recovery to 0.54. v3 holds self-image at +0.30 after the same first negative hit—a 0.51-point advantage over v2—and recovers to 0.69, a 28% improvement. Under alternating feedback, v3’s meta-cognition detects oscillation and autonomously raises the temporal decay rate, eliminating confusion states entirely and maintaining a persistent positive bias (mean +0.32 vs. v2’s +0.01). A critical failure case under sustained negative input reveals that meta cognitive rules, which bind to context, can amplify damage. All source code and data are released as open-source software.
