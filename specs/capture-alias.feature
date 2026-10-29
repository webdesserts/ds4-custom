Feature: Unified --scr server configuration and legacy flag compatibility
  Passing --scr (or a nonzero --scr-tail-tokens) enables the complete Suffix
  Cache Reuse production stack: logical coordinates, fast batch capture,
  planner-owned fresh-gap space padding, archive capacity equal to the context
  size, K defaulting to 32 (1..32), and a 5,000 ms planner budget. Historical
  selector flags are accepted quietly for launch compatibility, while the
  retired row-chunks flag is rejected before startup.

  Scenario: --scr enables the full production stack with default parameters
    Given a DeepSeek V4.1 Metal server configuration on loopback
    When the user supplies --scr without explicit tuning overrides
    Then SCR is enabled with archive capacity equal to the context size
    And max ranges defaults to 32 with a 5,000 ms planner budget
    And logical coordinates, fast batch capture, and planner fresh-gap space padding are active

  Scenario: --scr-max-ranges accepts K from 1 through 32 and rejects out-of-range bounds
    Given a valid --scr server configuration
    When the user supplies --scr-max-ranges K for any K in 1 through 32
    Then startup configuration validation accepts K
    And supplying K equal to 0 or greater than 32 is refused before listener or model startup

  Scenario: ds4.reuse.v1 records carry exactly the planned ranges
    Given a valid --scr server configuration with any K in 1 through 32
    When a request's reuse telemetry record is written
    Then plan.ranges holds exactly plan.range_count entries, from 0 up to K
    And no fixed-slot padding entries are emitted

  Scenario: historical launch flags remain accepted as compatibility no-ops
    Given the recorded production launch argv using --scr-tail-tokens, --scr-logical-coordinates, and --scr-capture-batch or --scr-capture-batch-trial
    When it is parsed at the server configuration boundary
    Then startup selects the unified production SCR stack without error

  Scenario: retired --scr-capture-row-chunks option is refused
    Given any server configuration
    When the user supplies --scr-capture-row-chunks
    Then parsing is refused before listener or model startup

  Scenario: --scr combines with --kv-disk-dir to enable the persistent SCR checkpoint pool
    Given a valid --scr server configuration
    When the user supplies --kv-disk-dir DIR and optional --kv-disk-space-mb N
    Then startup opens the persistent SCR checkpoint pool in DIR instead of the prefix-only KV cache
    And sealed archives are persisted across server restarts and branch switches

  Scenario: per-request reuse field selects between scr and exact modes
    Given a running --scr server
    When a chat completion request supplies "reuse": "scr" or omits "reuse"
    Then the server executes the SCR planner and multi-range reuse policy
    When a chat completion request supplies "reuse": "exact"
    Then the server bypasses the SCR planner and reuses only an exact live prefix
