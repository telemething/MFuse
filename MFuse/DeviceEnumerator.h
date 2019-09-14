#pragma once

#define USING_WINDOWS

#ifdef USING_WINDOWS

#include <Windows.h>
#include <dshow.h>

#pragma comment(lib, "strmiids")

#include <map>
#include <string>

/*struct Device {
	int id; // This can be used to open the device in OpenCV
	std::string devicePath;
	std::string deviceName; // This can be used to show the devices to the user
};*/

class DeviceEnumerator {

public:

	struct Device {
		int id; // This can be used to open the device in OpenCV
		std::string devicePath;
		std::string deviceName; // This can be used to show the devices to the user
	};

	DeviceEnumerator() = default;
	std::map<int, Device> getDevicesMap(const GUID deviceClass);
	std::map<int, Device> getVideoDevicesMap();
	std::map<int, Device> getAudioDevicesMap();

private:

	std::string ConvertBSTRToMBS(BSTR bstr);
	std::string ConvertWCSToMBS(const wchar_t* pstr, long wslen);

};

#endif //USING_WINDOWS

