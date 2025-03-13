# Makefile for Rust workspace
# Contains targets for the entire workspace and individual crates

# Define crates in the workspace
CRATES := loan synthetic_data ezkl

# Default target: build all workspace members
all:
	cargo build --workspace

# Build all workspace members in release mode
release:
	cargo build --workspace --release

# Check all workspace members
check:
	cargo check --workspace

# Test all workspace members
test:
	cargo test --workspace

# Clean all build artifacts
clean:
	cargo clean

# Run clippy on all workspace members
clippy:
	cargo clippy --workspace

# Run clippy with auto-fix on all workspace members
clippy-fix:
	cargo clippy --workspace --fix

# Individual crate targets
loan:
	cargo build -p loan

synthetic_data:
	cargo build -p synthetic_data

ezkl:
	cargo build -p ezkl

# Check individual crates
check-loan:
	cargo check -p loan

check-synthetic_data:
	cargo check -p synthetic_data

check-ezkl:
	cargo check -p ezkl

# Test individual crates
test-loan:
	cargo test -p loan

test-synthetic_data:
	cargo test -p synthetic_data

test-ezkl:
	cargo test -p ezkl

# Build individual crates in release mode
release-loan:
	cargo build -p loan --release

release-synthetic_data:
	cargo build -p synthetic_data --release

release-ezkl:
	cargo build -p ezkl --release

# Run clippy on individual crates
clippy-loan:
	cargo clippy -p loan

clippy-synthetic_data:
	cargo clippy -p synthetic_data

clippy-ezkl:
	cargo clippy -p ezkl

# Run clippy with auto-fix on individual crates
clippy-fix-loan:
	cargo clippy -p loan --fix

clippy-fix-synthetic_data:
	cargo clippy -p synthetic_data --fix

clippy-fix-ezkl:
	cargo clippy -p ezkl --fix

# Show help information
help:
	@echo "Available targets:"
	@echo "  all                  - Build all workspace members (default)"
	@echo "  check                - Check all workspace members"
	@echo "  test                 - Test all workspace members"
	@echo "  clean                - Clean all build artifacts"
	@echo "  release              - Build all workspace members in release mode"
	@echo "  loan                 - Build the loan crate"
	@echo "  synthetic_data       - Build the synthetic_data crate"
	@echo "  ezkl                 - Build the ezkl crate"
	@echo "  check-loan           - Check the loan crate"
	@echo "  check-synthetic_data - Check the synthetic_data crate"
	@echo "  check-ezkl           - Check the ezkl crate"
	@echo "  test-loan            - Test the loan crate"
	@echo "  test-synthetic_data  - Test the synthetic_data crate"
	@echo "  test-ezkl            - Test the ezkl crate"
	@echo "  release-loan         - Build the loan crate in release mode"
	@echo "  release-synthetic_data - Build the synthetic_data crate in release mode"
	@echo "  release-ezkl         - Build the ezkl crate in release mode"
	@echo "  clippy               - Run clippy on all workspace members"
	@echo "  clippy-fix           - Run clippy with auto-fix on all workspace members"
	@echo "  clippy-loan          - Run clippy on the loan crate"
	@echo "  clippy-synthetic_data - Run clippy on the synthetic_data crate"
	@echo "  clippy-ezkl          - Run clippy on the ezkl crate"
	@echo "  clippy-fix-loan      - Run clippy with auto-fix on the loan crate"
	@echo "  clippy-fix-synthetic_data - Run clippy with auto-fix on the synthetic_data crate"
	@echo "  clippy-fix-ezkl      - Run clippy with auto-fix on the ezkl crate"
	@echo "  help                 - Show this help information"

# Mark all targets as PHONY (not associated with files)
.PHONY: all check test clean release help \
	loan synthetic_data ezkl \
	check-loan check-synthetic_data check-ezkl \
	test-loan test-synthetic_data test-ezkl \
	release-loan release-synthetic_data release-ezkl \
	clippy clippy-fix \
	clippy-loan clippy-synthetic_data clippy-ezkl \
	clippy-fix-loan clippy-fix-synthetic_data clippy-fix-ezkl

