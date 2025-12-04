# Non-volatile memory simulator
# Pre-release version, r131

# Define directories
SRC_DIR := src/cpp
BUILD_DIR := build
TARGET := $(BUILD_DIR)/nvsim

# define tool chain
CXX := g++
RM := rm -f

# define build options
# compile options
CXXFLAGS := -Wall -I$(SRC_DIR)
# link options
LDFLAGS :=
# link librarires
LDLIBS :=

# construct list of .cpp and their corresponding .o and .d files
SRC := $(wildcard $(SRC_DIR)/*.cpp)
INC := 
DBG :=
OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
DEP := $(BUILD_DIR)/Makefile.dep

# file disambiguity is achieved via the .PHONY directive
.PHONY : all clean dbg nuke

all : $(TARGET)

dbg: DBG += -ggdb -g
dbg: $(TARGET)

$(TARGET) : $(OBJ) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DBG) $(INC) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean :
	$(RM) -r $(BUILD_DIR)

nuke :
	$(RM) -r $(BUILD_DIR)
	$(RM) -r plots
	$(RM) -r measurements/sierpinski_data
	$(RM) -r measurements/sierpinski_4_variants

depend $(DEP): | $(BUILD_DIR)
	@echo Makefile - creating dependencies for: $(SRC)
	@$(RM) $(DEP)
	@$(CXX) -E -MM $(INC) $(SRC) >> $(DEP)

ifeq (,$(findstring clean,$(MAKECMDGOALS)))
-include $(DEP)
endif
