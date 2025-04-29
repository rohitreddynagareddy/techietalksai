class BookingTool:
    def __init__(self):
        self.bookings = []

    def run(self, input_str: str) -> str:
        """Handle natural language booking requests"""
        try:
            # Try to parse structured input
            if "=" in input_str:
                params = dict(item.split("=") for item in input_str.split(","))
                return self._create_booking(params)
            
            # Handle natural language input
            return ("Please provide details in format: "
                    "'name=John Doe, date=2024-03-20, time=15:00, service=Massage'")
        
        except Exception as e:
            return f"Booking failed: {str(e)}"

    def _create_booking(self, params: dict) -> str:
        """Actual booking logic"""
        required = ["name", "date", "time", "service"]
        if not all(key in params for key in required):
            missing = [key for key in required if key not in params]
            return f"Missing parameters: {', '.join(missing)}"
        
        booking_id = len(self.bookings) + 1
        self.bookings.append({
            "id": booking_id,
            **params
        })
        return (f"Booking #{booking_id} confirmed!\n"
                f"Service: {params['service']}\n"
                f"Date: {params['date']} at {params['time']}\n"
                f"Client: {params['name']}")