
import asyncio

from winsdk.windows.devices.geolocation import (
    Geolocator,
    GeolocationAccessStatus,
)


async def main():
    permission = await Geolocator.request_access_async()

    if permission != GeolocationAccessStatus.ALLOWED:
        print("Windows chưa cấp quyền định vị:", permission)
        return

    locator = Geolocator()
    position = await locator.get_geoposition_async()

    coordinate = position.coordinate.point.position

    print("Latitude :", coordinate.latitude)
    print("Longitude:", coordinate.longitude)
    print("Accuracy :", position.coordinate.accuracy, "m")


asyncio.run(main())
