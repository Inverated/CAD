# Makefile for Roti Proa FreeCAD project

# Detect FreeCAD command (different on different systems)
FREECAD_APP := /Applications/FreeCAD.app/Contents/MacOS/FreeCAD
#FREECAD_APP := /Applications/FreeCAD.app/Contents/Resources/bin/freecad
FREECAD := $(shell which freecad 2>/dev/null || \
                   which freecadcmd 2>/dev/null || \
                   (test -f $(FREECAD_APP) && echo $(FREECAD_APP)) || \
                   echo "freecad")

FREECAD_CMD := $(FREECAD_APP) --console

# Directories
SRC_DIR := src
OUTPUT_DIR := output
RENDER_DIR := $(OUTPUT_DIR)/renders
EXPORT_DIR := $(OUTPUT_DIR)/exports

# Discover all boats and configurations dynamically
BOATS := $(basename $(notdir $(wildcard $(SRC_DIR)/boats/*.py)))
CONFIGS := $(basename $(notdir $(wildcard $(SRC_DIR)/configurations/*.py)))

# Filter out __pycache__ and backup files
BOATS := $(filter-out __pycache__,$(BOATS))
CONFIGS := $(filter-out __pycache__ default,$(CONFIGS))

# Which boat to build (RP2 or RP3)
BOAT ?= RP2
PARAMS := boats.$(BOAT)

# What configuration to use (CloseHaul etc)
CONFIG ?= CloseHaul
CONFIG_PARAM := configurations.$(CONFIG)

OUTPUT_NAME := RotiProa_$(BOAT)_$(CONFIG)

# Main macro
MACRO := $(SRC_DIR)/RotiProa.FCMacro

# Output files
FCSTD := $(OUTPUT_DIR)/$(OUTPUT_NAME).FCStd
STEP := $(EXPORT_DIR)/$(OUTPUT_NAME).step

# Default target - build all boats with all configurations
.PHONY: all
all:
	@echo "Building all boats with all configurations..."
	@echo "Boats: $(BOATS)"
	@echo "Configs: $(CONFIGS)"
	@$(foreach boat,$(BOATS),$(foreach config,$(CONFIGS),$(MAKE) build BOAT=$(boat) CONFIG=$(config);))
	@echo "All builds complete!"

# Create output directories
$(OUTPUT_DIR) $(RENDER_DIR) $(EXPORT_DIR):
	mkdir -p $@

.PHONY: build
build: $(OUTPUT_DIR)
	@echo "Building $(BOAT) with $(CONFIG) configuration..."
	@$(FREECAD_CMD) $(MACRO) $(PARAMS) $(CONFIG_PARAM)
	@echo "Setting visibility..."
	@$(SRC_DIR)/fix_visibility.sh $(OUTPUT_DIR)/RotiProa_$(BOAT)_$(CONFIG).FCStd $(FREECAD_APP) 2>&1 | grep -v "^[A-Z_]*=" | grep -v "3DconnexionNavlib" | grep -v "^Running:" || true
	@echo "Build complete!"

# Export to various formats (requires adding export commands to macro)
.PHONY: export
export: build $(EXPORT_DIR)
	@echo "Exporting model..."
# Add export commands here once you modify the macro
	@echo "Export complete!"

# Render images (requires adding render commands to macro)
.PHONY: render
render: build $(RENDER_DIR)
	@echo "Rendering images..."
# Add render commands here once you modify the macro
	@echo "Render complete!"

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning output files..."
	rm -rf $(OUTPUT_DIR)
	@echo "Clean complete!"

# Build specific boats with all configurations
.PHONY: rp2 rp3
rp2:
	@$(foreach config,$(CONFIGS),$(MAKE) build BOAT=RP2 CONFIG=$(config);)

rp3:
	@$(foreach config,$(CONFIGS),$(MAKE) build BOAT=RP3 CONFIG=$(config);)

# Build all boats (all configurations)
.PHONY: both boats
both boats: all

# Build specific configuration with all boats
.PHONY: closehaul beamreach broadreach goosewing
closehaul:
	@$(foreach boat,$(BOATS),$(MAKE) build BOAT=$(boat) CONFIG=CloseHaul;)

beamreach:
	@$(foreach boat,$(BOATS),$(MAKE) build BOAT=$(boat) CONFIG=BeamReach;)

broadreach:
	@$(foreach boat,$(BOATS),$(MAKE) build BOAT=$(boat) CONFIG=BroadReach;)

goosewing:
	@$(foreach boat,$(BOATS),$(MAKE) build BOAT=$(boat) CONFIG=GooseWing;)

# Show statistics
.PHONY: stats
stats: build
	@echo "Model statistics:"
	@python3 $(SRC_DIR)/stats.py 2>/dev/null || echo "Run build first"

# Help
.PHONY: help
help:
	@echo "Roti Proa Makefile"
	@echo ""
	@echo "Discovered boats: $(BOATS)"
	@echo "Discovered configurations: $(CONFIGS)"
	@echo ""
	@echo "Main targets:"
	@echo "  make             - Build ALL boats with ALL configurations"
	@echo "  make all         - Same as above"
	@echo "  make build       - Build single boat+config (BOAT=$(BOAT) CONFIG=$(CONFIG))"
	@echo ""
	@echo "Boat-specific targets:"
	@echo "  make rp2         - Build RP2 with all configurations"
	@echo "  make rp3         - Build RP3 with all configurations"
	@echo ""
	@echo "Configuration-specific targets:"
	@echo "  make closehaul   - Build all boats in CloseHaul configuration"
	@echo "  make beamreach   - Build all boats in BeamReach configuration"
	@echo "  make broadreach  - Build all boats in BroadReach configuration"
	@echo "  make goosewing   - Build all boats in GooseWing configuration"
	@echo ""
	@echo "Utility targets:"
	@echo "  make clean       - Remove all generated files"
	@echo "  make stats       - Show model statistics"
	@echo "  make check       - Check FreeCAD installation"
	@echo "  make help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make build BOAT=RP2 CONFIG=BeamReach"
	@echo "  make rp2"
	@echo "  make closehaul"
	@echo ""
	@echo "FreeCAD: $(FREECAD)"

# Check if FreeCAD is installed
.PHONY: check
check:
	@echo "Checking for FreeCAD..."
	@$(FREECAD) --version || (echo "FreeCAD not found!" && exit 1)
	@echo "FreeCAD found: $(FREECAD)"
