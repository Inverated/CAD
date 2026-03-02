class Load_Array():
    def __init__(self, circuit, components, constants, load_list: list):
        self.circuit = circuit
        self.components = components
        self.constants = constants
        self.sub_circuit = self.__create_array(load_list)
        self.terminal = None
        self.terminal_id = None
            
    def __create_array(self, load_list):
        self.terminal = f"total_load_in_voltage"
        self.terminal_id = f"total_load_in_current"
        
        for index, load in enumerate(load_list):
            load_name = load.name()
            load_pos = f"{load_name}_positive"
            load_neg = f"{load_name}_negative"
            
            if index == 0:
                self.circuit.V(self.terminal, load_pos, self.constants["GROUNDING_RESISTANCE"])
            else:
                prev_load_name = load_list[index-1].name()
                self.circuit.R(f"load_{index}_internal", load_pos, f"{prev_load_name}_positive", self.constants["WIRE_RESISTANCE"])
                    
            self.circuit.R(f"load_{index}_grounding", load_neg, self.circuit.gnd, self.constants["GROUNDING_RESISTANCE"])
    
    def get_terminal(self):
        return self.terminal
    
    def get_terminal_id(self):
        return self.terminal_id
    
    
    def __str__(self):
        return f"""{self.constants['BARF']}Load Array Setup{self.constants['BARE']}
Loads: {', '.join([load.name() for load in self.load_list])}
Count: {len(self.load_list)}
"""
