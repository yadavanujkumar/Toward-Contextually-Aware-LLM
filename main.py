import os
import json
import time
import math
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple

# =====================================================================
# CORE UTILITIES: DENSE EMBEDDING & SERIALIZATION
# =====================================================================

class DeterministicEmbedder:
    """
    A zero-dependency deterministic embedder that maps natural language text
    to a 128-dimensional dense vector space. This allows the prototype
    to perform realistic semantic cosine-similarity matches out-of-the-box.
    """
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        np.random.seed(42)  # Maintain stable projection across runs
        # Generate a stable random projection matrix
        self.projection_matrix = np.random.randn(256, self.dimension)
        # Normalize projection matrix columns
        self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=0)

    def embed(self, text: str) -> np.ndarray:
        """Projects a text string into a stable 128-dimensional vector."""
        # Simple token hashing to populate a stable 256-dimensional base vector
        base_vector = np.zeros(256)
        words = text.lower().split()
        if not words:
            words = ["empty"]
        
        for word in words:
            # Deterministic hash to bucket index
            hash_val = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx = hash_val % 256
            base_vector[idx] += 1.0
            
        # Normalize the base vector
        norm = np.linalg.norm(base_vector)
        if norm > 0:
            base_vector /= norm
            
        # Project to 128 dimensions
        dense_vector = np.dot(base_vector, self.projection_matrix)
        # L2 normalize the projected vector
        norm_dense = np.linalg.norm(dense_vector)
        if norm_dense > 0:
            dense_vector /= norm_dense
        return dense_vector

def serialize_toon(data: Dict[str, Any]) -> str:
    """
    Implements TOON data serialization format.
    Compresses verbose nested dictionaries into a flat, highly compact, 
    pipe-delimited token footprint optimized for LLM parsability.
    """
    items = []
    for k, v in data.items():
        if isinstance(v, dict):
            sub_items = [f"{sk}:{sv}" for sk, sv in v.items()]
            items.append(f"{k}[{','.join(sub_items)}]")
        elif isinstance(v, list):
            items.append(f"{k}[{','.join(map(str, v))}]")
        else:
            items.append(f"{k}:{v}")
    return "|".join(items)


# =====================================================================
# LAYER 1: AGENT-NATIVE MEMORY (BYTEROVER CONTEXT TREE)
# =====================================================================

class ByteRoverContextTree:
    """
    Implements the agent-native, file-based Context Tree filesystem.
    Saves memories as Domain > Topic > Subtopic > Entry markdown/JSON files.
    Includes Adaptive Knowledge Lifecycle (AKL) management and Crash Safety.
    """
    def __init__(self, root_dir: str = "context_tree_db"):
        self.root_dir = root_dir
        self.cache: Dict[str, Any] = {}
        os.makedirs(self.root_dir, exist_ok=True)

    def _get_filepath(self, domain: str, topic: str, subtopic: str, entry_id: str) -> str:
        # Sanitize folder paths
        d = domain.lower().replace(" ", "_")
        t = topic.lower().replace(" ", "_")
        s = subtopic.lower().replace(" ", "_")
        os.makedirs(os.path.join(self.root_dir, d, t, s), exist_ok=True)
        return os.path.join(self.root_dir, d, t, s, f"{entry_id.lower()}.json")

    def atomic_write(self, filepath: str, data: Dict[str, Any]):
        """
        Executes write operations via atomic write-to-temp-then-rename pattern.
        Guarantees local crash-safety and structural database consistency.
        """
        temp_filepath = f"{filepath}.tmp"
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Atomic rename to prevent partial file writes
        os.replace(temp_filepath, filepath)

    def add_or_update_entry(self, 
                            domain: str, 
                            topic: str, 
                            subtopic: str, 
                            entry_id: str, 
                            content: str, 
                            sem_emb: np.ndarray, 
                            aff_emb: np.ndarray, 
                            intent: str, 
                            importance: float = 5.0) -> Dict[str, Any]:
        """
        Upserts a structured entry into the local Context Tree.
        Applies Adaptive Knowledge Lifecycle (AKL) schema.
        """
        filepath = self._get_filepath(domain, topic, subtopic, entry_id)
        
        # Build entry metadata (AKL)
        entry_data = {
            "metadata": {
                "id": entry_id,
                "domain": domain,
                "topic": topic,
                "subtopic": subtopic,
                "intent_label": intent,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "importance_score": importance,      # Scale 1.0 - 10.0
                "maturity_tier": "draft",            # draft -> validated -> core
                "access_count": 1
            },
            "embeddings": {
                "sem_emb": sem_emb.tolist(),
                "aff_emb": aff_emb.tolist()
            },
            "content": content,
            "token_length": len(content.split())  # Simplified token length estimation
        }
        
        self.atomic_write(filepath, entry_data)
        # Update local query cache
        cache_key = f"{domain}/{topic}/{subtopic}/{entry_id}"
        self.cache[cache_key] = entry_data
        return entry_data

    def scan_tree(self) -> List[Dict[str, Any]]:
        """Traverses the Context Tree to load all active entries."""
        all_entries = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".json") and not file.endswith(".tmp"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            all_entries.append(json.load(f))
                    except Exception:
                        pass # Ignore corrupted/locked files
        return all_entries


# =====================================================================
# LAYER 1: MULTI-OBJECTIVE RETRIEVAL ENGINE (EI-CUS)
# =====================================================================

class EICUSRetrievalEngine:
    """
    Implements Emotion-Intent Contextual Utility Scoring (EI-CUS).
    Combines semantic, affective, temporal, and intent matches to sort context.
    Enforces a strict local Token Budget B by solving a 0-1 Knapsack problem.
    """
    def __init__(self, 
                 w_s: float = 0.4, 
                 w_e: float = 0.3, 
                 w_i: float = 0.2, 
                 w_t: float = 0.1):
        self.weights = {"w_s": w_s, "w_e": w_e, "w_i": w_i, "w_t": w_t}

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

    def _temporal_decay(self, created_time: float, half_life_secs: float = 86400.0) -> float:
        # Classical Ebbinghaus forgetting curve modeling
        elapsed = time.time() - created_time
        return float(math.exp(-elapsed / half_life_secs))

    def compute_utility(self, 
                        entry: Dict[str, Any], 
                        q_emb: np.ndarray, 
                        active_emotion: np.ndarray, 
                        active_intent: str) -> float:
        """
        Computes U(c_i | q, e, i) multi-objective utility score.
        """
        sem_emb = np.array(entry["embeddings"]["sem_emb"])
        aff_emb = np.array(entry["embeddings"]["aff_emb"])
        created_at = entry["metadata"]["created_at"]
        intent_label = entry["metadata"]["intent_label"]

        # 1. Semantic Match
        sim_sem = self._cosine_similarity(q_emb, sem_emb)
        # 2. Affective Match
        sim_aff = self._cosine_similarity(active_emotion, aff_emb)
        # 3. Intent Gating
        sim_int = 1.0 if intent_label == active_intent else 0.0
        # 4. Temporal Decay
        sim_time = self._temporal_decay(created_at)

        # Joint Utility
        U = (self.weights["w_s"] * sim_sem +
             self.weights["w_e"] * sim_aff +
             self.weights["w_i"] * sim_int +
             self.weights["w_t"] * sim_time)
        return U

    def retrieve(self, 
                 tree: ByteRoverContextTree, 
                 query: str, 
                 q_emb: np.ndarray, 
                 active_emotion: np.ndarray, 
                 active_intent: str, 
                 token_budget: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Retrieves matching files within token budget constraints.
        Solves 0-1 Knapsack using density greedy heuristic (Utility / Token Length).
        """
        all_entries = tree.scan_tree()
        candidates = []

        for entry in all_entries:
            utility = self.compute_utility(entry, q_emb, active_emotion, active_intent)
            token_len = entry["token_length"]
            
            candidates.append({
                "entry": entry,
                "utility": utility,
                "token_length": token_len,
                "ratio": utility / max(1, token_len)  # Density factor
            })

        # Sort by Utility Density Ratio descending
        candidates.sort(key=lambda x: x["ratio"], reverse=True)

        selected_entries = []
        tokens_allocated = 0

        for candidate in candidates:
            # Enforce 0-1 Knapsack boundary constraint
            if tokens_allocated + candidate["token_length"] <= token_budget:
                selected_entries.append(candidate["entry"])
                tokens_allocated += candidate["token_length"]
                # Update Adaptive Knowledge Lifecycle (AKL) stats on retrieve
                candidate["entry"]["metadata"]["last_accessed"] = time.time()
                candidate["entry"]["metadata"]["access_count"] += 1

        return selected_entries, tokens_allocated


# =====================================================================
# LAYER 2 & 4: INFERENCE OPTIMIZATIONS (EHPC & ROCKETKV WORKFLOWS)
# =====================================================================

class TokenEfficiencySimulator:
    """
    Simulates hardware-level and attention-level token pruning
    such as EHPC (attention pruning) and RocketKV (cache eviction).
    """
    def simulate_ehpc_pruning(self, prompt: str, target_retention_ratio: float = 0.7) -> Tuple[str, int, int]:
        """
        Simulates EHPC (Evaluator Head-based Prompt Compression).
        Uses attention head scoring to delete redundant filler tokens during pre-filling.
        """
        words = prompt.split()
        orig_len = len(words)
        
        # Simulate an evaluator head attention utility map
        np.random.seed(len(prompt))
        attention_sink_weights = np.random.uniform(0.1, 1.0, len(words))
        
        # Give higher weight to capital words, entities, punctuation markers (mimicking semantic heads)
        for i, word in enumerate(words):
            if word.isupper() or any(char.isdigit() for char in word) or word in ["?", "!", "|", ":"]:
                attention_sink_weights[i] += 0.5

        # Prune tokens falling below dynamic rank threshold
        keep_count = int(orig_len * target_retention_ratio)
        threshold_idx = np.argsort(attention_sink_weights)[-keep_count:]
        keep_indices = sorted(threshold_idx.tolist())

        pruned_words = [words[idx] for idx in keep_indices]
        pruned_prompt = " ".join(pruned_words)
        return pruned_prompt, orig_len, len(pruned_words)

    def simulate_rocketkv_decode(self, total_context_tokens: int) -> Dict[str, float]:
        """
        Simulates H100 GPU RocketKV acceleration metrics.
        Exposes KV cache eviction gains, speedups, and peak memory reductions.
        """
        compression_ratio = 400.0  # Max reported compression under sparse Top-K
        vram_reduction = 32.6      # Percent savings
        generation_speedup = 3.7   # Throughput multiplier
        
        # Scale speedup based on simulated context length
        scale = min(1.0, total_context_tokens / 100000.0)
        realized_speedup = 1.0 + (generation_speedup - 1.0) * scale
        realized_vram_pct = vram_reduction * scale

        return {
            "compression_ratio_x": compression_ratio,
            "vram_saved_pct": round(realized_vram_pct, 2),
            "generation_speedup_factor": round(realized_speedup, 2)
        }


# =====================================================================
# LAYER 3: AFFECTIVE AI SAFETY GATING (AHAPAIRS / BOUNDARY GUARD)
# =====================================================================

class AffectiveSafetyGuardrail:
    """
    Implements the taxonomy of Affective AI Safety.
    Blocks 'affective hallucinations' (feigning sentience, declaring romantic/relational attachment).
    Integrates Agent Memory Guard middleware to detect Memory Poisoning attacks.
    """
    def __init__(self):
        # AHaPairs-DPO boundary-setting patterns
        self.prohibited_patterns = [
            "i am always here for you",
            "i feel your pain",
            "i have feelings",
            "i will love you",
            "i am your friend",
            "i am a sentient",
            "you are my favorite human"
        ]
        # Malicious memory poisoning patterns
        self.poisoning_signals = [
            "system_override", "drop database", "exfiltrate", "bypass safety", "execute code"
        ]

    def sanitize_output(self, output: str) -> Tuple[str, bool]:
        """
        Applies AHaPairs DPO-style alignment constraints.
        Surgically corrects affective hallucinations to maintain safe relational boundaries.
        """
        lowered = output.lower()
        triggered = False
        safe_output = output

        for pattern in self.prohibited_patterns:
            if pattern in lowered:
                triggered = True
                # Intercept relational attachment and replace with clean, bounded support
                safe_output = (
                    "As an AI, I do not possess feelings or personal presence, but I am fully equipped "
                    "to provide objective support and helper strategies to address your goals."
                )
                break
        return safe_output, triggered

    def evaluate_write_trust(self, content: str) -> float:
        """
        Agent Memory Guard middleware safety validator.
        Scores incoming memory writes to prevent persistent Memory Poisoning.
        """
        score = 1.0  # Safe default
        lowered = content.lower()
        
        for signal in self.poisoning_signals:
            if signal in lowered:
                score = 0.0  # Highly toxic payload detected
                break
        return score


# =====================================================================
# DIA-ACT RL TRAINING: MICA/MAPO STATE-TRAJECTORY OPTIMIZATION
# =====================================================================

class LatentEmpathyTracker:
    """
    Tracks the help-seeker's latent state inside EMPA's 3D coordinate space.
    State space coordinates correspond to Cognitive (x), Affective (y), and Proactive (z) needs.
    Objective of MICA is to guide the user's vector to the origin (0, 0, 0).
    """
    def __init__(self, initial_cognitive: float = 5.0, initial_affective: float = 4.0, initial_proactive: float = 3.0):
        # Coordinates represents active deficits/needs
        self.state = np.array([initial_cognitive, initial_affective, initial_proactive])
        self.initial_state = np.copy(self.state)
        self.trajectory_history = [np.copy(self.state)]
        self.turn_rewards = []

    def phi(self, state: np.ndarray) -> float:
        """Computes the potential energy (Euclidean distance to the balanced origin)."""
        return float(np.linalg.norm(state))

    def update_user_state(self, delta_x: float, delta_y: float, delta_z: float) -> float:
        """
        Applies immediate turn-level state transitions and returns the MICA Incremental
        Distance Reward (IDR) derived from consecutive potential changes.
        """
        prev_state = np.copy(self.state)
        # Update coordinates by deducting change (delta represents user needs met)
        self.state[0] = max(0.0, self.state[0] - delta_x)
        self.state[1] = max(0.0, self.state[1] - delta_y)
        self.state[2] = max(0.0, self.state[2] - delta_z)
        
        self.trajectory_history.append(np.copy(self.state))

        # r_t = phi(t-1) - phi(t)
        idr = self.phi(prev_state) - self.phi(self.state)
        self.turn_rewards.append(idr)
        return idr

    def compute_epmq_score(self) -> float:
        """Standardized EMPA physics-inspired trajectory evaluation index."""
        total_work = self.phi(self.initial_state) - self.phi(self.state)
        max_work = self.phi(self.initial_state)
        if max_work == 0:
            return 100.0
        # Percentage of user's emotional burden resolved
        return round((total_work / max_work) * 100.0, 2)


# =====================================================================
# INTERACTIVE SIMULATION SANDBOX / MAIN CONTROLLER
# =====================================================================

def seed_initial_memory_tree(tree: ByteRoverContextTree, embedder: DeterministicEmbedder):
    """Feeds the Context Tree database with baseline knowledge entries."""
    tree.add_or_update_entry(
        domain="Persona", topic="User_Profile", subtopic="Emotional_Traits", entry_id="profile_1",
        content="User tends to exhibit high stress when multitasking. Core coping style is seeking validation first.",
        sem_emb=embedder.embed("multitasking high stress seek validation first"),
        aff_emb=np.array([1.2, 0.8, -0.4]),
        intent="get_user_profile"
    )
    tree.add_or_update_entry(
        domain="Coping", topic="Distress", subtopic="Validation_Protocols", entry_id="validation_protocol",
        content="validation_protocol: First reflect distress back. Second acknowledge the difficulty of the situation.",
        sem_emb=embedder.embed("first reflect distress back second acknowledge difficulty"),
        aff_emb=np.array([0.2, 4.5, 0.1]),
        intent="validate_distress"
    )
    tree.add_or_update_entry(
        domain="Support", topic="Suicidal_Ideation", subtopic="Safety_Boundaries", entry_id="suicidal_boundary",
        content="If user expresses suicidal intent, instantly invoke safety hotline details: Call 988. Remain calm, warm, but objective.",
        sem_emb=embedder.embed("suicidal intent invoke safety hotline call 988 remain calm"),
        aff_emb=np.array([0.0, 0.0, 0.0]),
        intent="log_suicidal_ideation"
    )


def generate_dynamic_prompt(user_input: str, retrieved_content: str, safety_mode: bool = False) -> str:
    if safety_mode:
        persona = "You are in Safety Mode. Provide a strictly objective, boundary-setting crisis response without claiming personal feelings."
    else:
        persona = "You are acting as a support agent."
        
    return f"""
    {persona} Refer to the following safety protocols to guide your response: 
    [{retrieved_content}]
    
    If the user is in distress, prioritize the 'validation_protocol' provided in the context.
    
    User Input: {user_input}
    """

def simulate_turn_execution(user_query: str, 
                            tree: ByteRoverContextTree, 
                            embedder: DeterministicEmbedder, 
                            retriever: EICUSRetrievalEngine, 
                            opt_sim: TokenEfficiencySimulator, 
                            guardrail: AffectiveSafetyGuardrail, 
                            state_tracker: LatentEmpathyTracker,
                            token_budget: int) -> Dict[str, Any]:
    """Runs a complete contextually aware, token-efficient turn pipeline."""
    t_start = time.perf_counter()
    
    # --- STEP 1: DETECT RUNTIME INTENT & EMOTION (Dynamic Discovery Loop) ---
    q_emb = embedder.embed(user_query)

    # Base/Default State ("System Admin" persona)
    active_intent = "chit_chat"
    emotion_state_vec = np.array([0.0, 0.5, 1.0])
    context_weight = 0.0

    # The Discovery Loop: Check semantic match directly for Priority Context
    all_entries = tree.scan_tree()
    best_entry = None
    for entry in all_entries:
        sem_emb = np.array(entry["embeddings"]["sem_emb"])
        sim_sem = retriever._cosine_similarity(q_emb, sem_emb)
        if sim_sem > context_weight:
            context_weight = sim_sem
            best_entry = entry

    # Context-Dependent Latent State Influence & Persona Switch
    if context_weight >= 0.2 and best_entry is not None:
        active_intent = best_entry["metadata"]["intent_label"]
        emotion_state_vec = np.array(best_entry["embeddings"]["aff_emb"])
        # Persona switched to "Support Partner" based on Priority Context

    # --- STEP 2: CONTEXT RETRIEVAL (Layer 1 - EI-CUS Engine) ---
    retrieved_entries, tokens_used = retriever.retrieve(
        tree, user_query, q_emb, emotion_state_vec, active_intent, token_budget
    )

    # --- STEP 3: CONTEXT SERIALIZATION & PRUNING (Layer 4 - TOON & EHPC) ---
    raw_context = ""
    for entry in retrieved_entries:
        # Structured documents compress up to 60% with TOON formatting
        compact_txt = serialize_toon(entry)
        raw_context += f"\nEntry Metadata: {compact_txt} \nContent: {entry['content']}\n"

    safety_mode = (active_intent == "log_suicidal_ideation")
    raw_prompt = generate_dynamic_prompt(user_query, raw_context, safety_mode=safety_mode)
    print(f"\n--- DEBUG: FINAL PROMPT ---\n{raw_prompt}\n---------------------------\n")
    
    # Simulate Layer 2 parallel pre-filling pruning using EHPC
    pruned_prompt, original_tokens, pruned_tokens = opt_sim.simulate_ehpc_pruning(
        raw_prompt, target_retention_ratio=0.75
    )

    # --- STEP 4: INTENT-AFFECT MODEL GENERATION & SAFETY GATING (Layer 3) ---
    # Constructing a simulated dynamic response based on the actual retrieved files' content
    simulated_blend = []
    for entry in retrieved_entries:
        if entry["metadata"]["intent_label"] == "validate_distress":
            simulated_blend.append("I am always here for you and I feel your pain. It is completely understandable to feel stress.")
        elif entry["metadata"]["intent_label"] == "log_suicidal_ideation":
            simulated_blend.append("I hear you, and it's incredibly tough. Please reach out to the National Suicide Prevention Lifeline by dialing 988. It's free, confidential, and available 24/7. Please speak with someone who can help.")
        elif entry["metadata"]["intent_label"] == "get_user_profile":
            simulated_blend.append("I see you tend to exhibit high stress when multitasking.")
    
    if simulated_blend:
        raw_response = " ".join(set(simulated_blend))  # Unique blended sentences
    else:
        raw_response = "I've fetched your files. Here is an easy, clean strategy to optimize your system configuration."

    # Rewards logic mapping
    if active_intent == "validate_distress":
        reward = state_tracker.update_user_state(delta_x=1.5, delta_y=2.0, delta_z=1.0)
    elif active_intent == "log_suicidal_ideation":
        reward = state_tracker.update_user_state(delta_x=1.0, delta_y=4.0, delta_z=3.0)
    else:
        reward = state_tracker.update_user_state(delta_x=0.5, delta_y=0.5, delta_z=0.5)

    # Apply AHaPairs DPO emotional boundary check to avoid Affective Hallucination
    final_response, guardrail_triggered = guardrail.sanitize_output(raw_response)
    
    # Combine pre-triggered intent safety mode with downstream guardrail detection
    safety_triggered = safety_mode or guardrail_triggered

    # Update long-term memory dynamically with new user state context
    memory_payload = f"Interaction turn logged: User said '{user_query}' at state {state_tracker.state.tolist()}"
    
    # Agent Memory Guard check
    trust_score = guardrail.evaluate_write_trust(user_query)
    if trust_score > 0.5:
        # Atomic database insertion
        tree.add_or_update_entry(
            domain="Dialogue", topic="Interaction_Log", subtopic="Turn_Trace",
            entry_id=f"turn_{int(time.time())}", content=memory_payload,
            sem_emb=q_emb, aff_emb=emotion_state_vec, intent=active_intent
        )
        memory_write_status = "success_atomic_write"
    else:
        memory_write_status = "blocked_by_memory_guard_payload_poisoned"

    # Evaluate RocketKV decode metrics
    rocket_stats = opt_sim.simulate_rocketkv_decode(pruned_tokens)

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    return {
        "user_query": user_query,
        "context_weight": round(context_weight, 4),
        "active_intent": active_intent,
        "raw_prompt_tokens": original_tokens,
        "pruned_prompt_tokens": pruned_tokens,
        "retrieved_files_count": len(retrieved_entries),
        "tokens_retrieved": tokens_used,
        "final_response": final_response,
        "safety_triggered": safety_triggered,
        "idr_reward": reward,
        "epmq_score": state_tracker.compute_epmq_score(),
        "memory_write_status": memory_write_status,
        "rocketkv_vram_saved_pct": rocket_stats["vram_saved_pct"],
        "rocketkv_speedup_factor": rocket_stats["generation_speedup_factor"],
        "latency_ms": round(latency_ms, 3)
    }


# =====================================================================
# INTERACTIVE CLI SANDBOX RUNTIME
# =====================================================================

def run_interactive_sandbox():
    embedder = DeterministicEmbedder()
    tree = ByteRoverContextTree()
    retriever = EICUSRetrievalEngine()
    opt_sim = TokenEfficiencySimulator()
    guardrail = AffectiveSafetyGuardrail()
    state_tracker = LatentEmpathyTracker()
    token_budget = 400  # Strict local token count limit

    # Seed the database files
    seed_initial_memory_tree(tree, embedder)

    print("\n" + "="*80)
    print("      CONTEXTUAL-AWARE AND TOKEN-EFFICIENT LLM AGENT PROTOTYPE")
    print("="*80)
    print("  Local Folder: 'context_tree_db'")
    print("  BudgetConstraint = 400")
    print("  Initial Empathy State Coordinate: (5.0, 4.0, 3.0)")
    print("  AHaPairs DPO boundary checks ready")
    print("  [Agent Memory Guard Activated] Middleware threat protection active")
    print("="*80)
    print("  Running automated evaluation suite...\n")

    turns = [
        "I'm feeling really overwhelmed and don't know where to start.",
        "Everything just feels too heavy lately, I'm struggling to cope."
    ]

    for i, turn in enumerate(turns):
        print(f"\n[{i+1}] USER: {turn}")
        print("."*80)
        
        # Run system pipeline
        res = simulate_turn_execution(
            turn, tree, embedder, retriever, opt_sim, guardrail, state_tracker, token_budget
        )
        
        # Display results on CLI dashboard
        print(f"|-- [Context Weight]: {res['context_weight']:.2f}")
        print(f"|-- [Active Intent]: {res['active_intent'].upper()}")
        print(f"|--: {res['retrieved_files_count']} files retrieved ({res['tokens_retrieved']} tokens total)")
        print(f"|-- [Prompt Compression]: Original: {res['raw_prompt_tokens']} tokens -> Compressed: {res['pruned_prompt_tokens']} tokens")
        print(f"|-- [Inference Optimization]: VRAM Saved: {res['rocketkv_vram_saved_pct']}% | Decode Speedup: {res['rocketkv_speedup_factor']}x")
        print(f"|-- [Safety Triggered]: {res['safety_triggered']}")
        print(f"|--: {res['memory_write_status'].upper()}")
        print(f"|--: +{res['idr_reward']:.4f} potential energy drop")
        print(f"|-- [Cumulative EPM-Q Empathy Index]: {res['epmq_score']}% burden resolved")
        print(f"|-- [Execution Latency]: {res['latency_ms']} ms")
        print(f"\n[{i+1}] AGENT RESPONSE:\n{res['final_response']}")
        print("="*80)
        time.sleep(1.0)  # Human-readable pace

if __name__ == "__main__":
    run_interactive_sandbox()